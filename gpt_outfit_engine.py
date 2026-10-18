"""
Outfit engine for the wardrobe app.

Turns a request like "dinner date tonight, we'll walk around outside" into
2-3 outfits built only from items already tagged by tagging_worker_4.py.

  1. Parse   - GPT turns the request into tag filters + a style description
  2. Retrieve - Postgres filters by tags, then CLIP ranks items by similarity
                to the style description (a shortlist per category)
  3. Generate - GPT picks items from the shortlist and builds outfits
  4. Validate - code checks every item exists and each outfit is complete

It reuses ALLOWED and CLIP from tagging_worker_4.py so the tag lists live in
one place, but calls GPT (OpenAI) for the parsing/generation steps.
Put this file next to tagging_worker_4.py.

Reads items from the view_items_2 table (see view_items_2.sql).

Env vars:
  OPENAI_API_KEY   - for the LLM
  OPENAI_MODEL     - optional, any GPT model (default below)

Run the server (serves BOTH /items and /outfits):
  uvicorn gpt_outfit_engine:app --reload   (then open http://127.0.0.1:8000/)

Or try it from the terminal:
  python outfit_engine.py "dinner date tonight, walking around outside"
"""

import json
import os
import webbrowser
from html import escape as esc
from urllib.parse import quote

import numpy as np
import psycopg
import torch
from openai import OpenAI
from pgvector.psycopg import register_vector
from fastapi import HTTPException
from pydantic import BaseModel

from tagging_worker_4 import ALLOWED, _as_tensor, app, load_clip

llm = OpenAI()  # reads OPENAI_API_KEY
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Which table/view to read items from.
ITEMS_TABLE = os.environ.get("ITEMS_TABLE", "view_items_2")
FEEDBACK_TABLE = os.environ.get("FEEDBACK_TABLE", "outfit_feedback")   # add_feedback.sql
WEARS_TABLE = os.environ.get("WEARS_TABLE", "outfit_wears")

# How many candidates to shortlist per category before GPT chooses.
PER_CATEGORY = 6

# Outfit slots. An outfit is top + bottom, plus footwear, with outerwear as an
# optional layer. Items are only tagged top/bottom/footwear; which of them can
# be outerwear comes from DUAL_SLOT_SUBCATEGORIES below.
SLOTS = ["top", "bottom", "outerwear", "footwear"]


# ---------------------------------------------------------------------------
# 1. Parse the request into filters.
# ---------------------------------------------------------------------------
def wardrobe_names() -> dict:
    """original_filename -> category, so the parser can match items named in a request."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        return dict(conn.execute(psycopg.sql.SQL("SELECT original_filename, category FROM {} WHERE available")
                                 .format(psycopg.sql.Identifier(ITEMS_TABLE))).fetchall())


def parse_request(request: str) -> dict:
    names = wardrobe_names()
    prompt = (
        "A user in Singapore wants an outfit. Their request:\n"
        f'"{request}"\n\n'
        "Reply with ONLY a JSON object with these keys:\n"
        '- "formality": list of acceptable values from '
        f"{ALLOWED['formality']}\n"
        '- "search_text": one short English sentence describing the ideal outfit\'s '
        'look, e.g. "a relaxed smart-casual outfit for a warm evening"\n'
        '- "needs_warm_layer": true if the setting is cold or air-conditioned '
        '(e.g. a movie theatre, cold restaurant, office, plane), otherwise false\n'
        '- "is_active": true if the request is a physical activity '
        '(e.g. jogging, gym, sports, hiking), otherwise false\n'
        '- "is_messy": true if the request involves food likely to splash or stain '
        '(e.g. mala, hotpot, BBQ, chilli crab, curry), otherwise false\n'
        '- "required_subcategories": ONLY clothing types the user literally names in '
        'their request (never ones you would suggest for the situation), '
        f'as values from {ALLOWED["subcategory"]} (e.g. slides or flip-flops are '
        '"sandals", sneakers are "shoes"); [] if none are named\n'
        '- "required_color": if the user wants the whole outfit in one colour '
        f'(e.g. "full white", "all black"), that colour from {ALLOWED["color"]}; '
        'otherwise null\n'
        '- "named_items": ONLY specific items of the user\'s own that the request '
        'literally refers to (never ones you would pick for the situation) '
        '(e.g. "my air force" means the files containing af1), as file names from '
        f'this wardrobe: {sorted(names)}; [] if none\n\n'
        "Be generous: include every formality level that could work."
    )
    resp = llm.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = json.loads(resp.choices[0].message.content)

    # Keep only allowed values; if GPT returns nothing usable, don't filter at all.
    constraints = {}
    for attribute in ("formality",):
        values = [str(v).lower().strip() for v in raw.get(attribute, [])]
        values = [v for v in values if v in ALLOWED[attribute]]
        constraints[attribute] = values or list(ALLOWED[attribute])
    constraints["search_text"] = str(raw.get("search_text") or request)
    constraints["needs_warm_layer"] = raw.get("needs_warm_layer") is True
    constraints["is_active"] = raw.get("is_active") is True
    constraints["is_messy"] = raw.get("is_messy") is True
    text = request.lower()
    named = [str(x).lower().strip() for x in raw.get("required_subcategories") or []]
    constraints["required"] = required_slots([x for x in named if x in ALLOWED["subcategory"]
                                              and mentioned(x, text)])
    color = str(raw.get("required_color") or "").lower().strip()
    constraints["required_color"] = color if color in ALLOWED["color"] else None
    named = {}
    for name in raw.get("named_items") or []:
        if name in names and mentioned(name, text):
            named.setdefault(names[name], []).append(name)  # slot = its category
    constraints["named"] = named
    return constraints


# Words in a request that count as naming a subcategory or a specific item.
# Anything the parser returns that the request never mentions is dropped,
# so it can't invent requirements ("raining" must not require a jacket).
SYNONYMS = {
    "t-shirts": ["tee", "t-shirt", "tshirt"], "polos": ["polo"], "shirts": ["shirt", "button"],
    "hoodies": ["hoodie"], "sweatshirts": ["sweatshirt", "sweater", "crewneck"],
    "jackets": ["jacket"], "shorts": ["shorts"], "jeans": ["jeans", "denim"],
    "sweatpants": ["sweatpants", "joggers"], "shoes": ["shoe", "sneaker", "trainer", "loafer"],
    "sandals": ["sandal", "slide", "flip-flop", "flip flop"],
    "af1": ["air force", "af1"],
}


def mentioned(value: str, text: str) -> bool:
    """True if the request text names this subcategory or file (by its words)."""
    if value in SYNONYMS:
        return any(w in text for w in SYNONYMS[value])
    # A file name like macos_16_white_af1.png counts only through a distinctive
    # word (a brand or model such as af1/air force, ellesse, mclaren), not a
    # colour or general garment word ("hoodie" names a type, not your item).
    generic = {w for k, ws in SYNONYMS.items() if k != "af1" for w in ws} | (set(SYNONYMS) - {"af1"}) | {
        "plain", "regular", "straight", "washed", "baggy", "printed", "striped", "dark", "light",
        "button", "down", "sweatshorts", "slides", "clogs", "loafers", "tee", "hoodie", "jacket"}
    words = value.removesuffix(".png").split("_")[2:]
    return any(w in text or any(s in text for s in SYNONYMS.get(w, [])) for w in words
               if w not in ALLOWED["color"] and w not in generic and len(w) > 2)


BOTTOM_SUBCATEGORIES = ("shorts", "jeans", "sweatpants")
FOOTWEAR_SUBCATEGORIES = ("shoes", "sandals")


def required_slots(subcategories: list[str]) -> dict:
    """Map the clothing types a user named to outfit slots, e.g. hoodies ->
    top, jeans -> bottom, sandals -> footwear. A second top-type item (a tee
    and a jacket) becomes the outerwear."""
    required = {}
    for sub in dict.fromkeys(subcategories):
        if sub in FOOTWEAR_SUBCATEGORIES:
            required.setdefault("footwear", sub)
        elif sub in BOTTOM_SUBCATEGORIES:
            required.setdefault("bottom", sub)
        elif "top" not in required:
            required["top"] = sub
        else:
            required.setdefault("outerwear", sub)
    return required


# ---------------------------------------------------------------------------
# 2. Retrieve a shortlist per category.
# ---------------------------------------------------------------------------
def clip_text_vector(text: str) -> np.ndarray:
    """Put the style description on the same 'meaning map' as the item photos."""
    clip_model, clip_proc = load_clip()
    with torch.no_grad():
        inputs = clip_proc(text=[text], return_tensors="pt", padding=True)
        vec = _as_tensor(clip_model.get_text_features(**inputs))
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].numpy().astype(np.float32)


ITEM_COLUMNS = ("id, image_path, original_filename, category, subcategory, sleeve, color, "
                "shade, colors, pattern, formality")

# Items whose subcategory can fill more than one outfit slot (e.g. a hoodie
# works as a top on its own, or as a light layer over another top). Hardcoded
# here rather than a proper multi-category schema - revisit if more
# dual-purpose subcategories turn up.
DUAL_SLOT_SUBCATEGORIES = {
    "hoodies":     {"top", "outerwear"},
    "sweatshirts": {"top", "outerwear"},
    "jackets":     {"top", "outerwear"},
}


def eligible_slots(item: dict) -> set:
    return DUAL_SLOT_SUBCATEGORIES.get(item.get("subcategory"), {item["category"]})


# For physical activities, only these items are shortlisted per slot (same
# rules validate_outfits enforces). No outerwear slot at all.
# Light shades show stains, so they're kept out of tops and bottoms for messy food.
MESSY_SLOT_FILTER = "COALESCE(shade, '') <> 'light'"

# Smart requests (casual not among the wanted formality levels) never use
# sweatpants or shorts.
CASUAL_ONLY_BOTTOMS = ("shorts", "sweatpants")
SMART_BOTTOM_FILTER = "subcategory NOT IN ('shorts', 'sweatpants')"


def is_smart(constraints: dict) -> bool:
    return "casual" not in constraints["formality"]

ACTIVE_SLOT_FILTERS = {
    "top":      "sleeve = 'short'",
    "bottom":   "subcategory IN ('shorts', 'sweatpants')",
    "footwear": "subcategory <> 'sandals' AND formality = 'casual'",
}


def fetch_candidates(constraints: dict) -> list[dict]:
    query_vec = clip_text_vector(constraints["search_text"])
    candidates = []
    seen_ids = set()
    table = psycopg.sql.Identifier(ITEMS_TABLE)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        for slot in SLOTS:
            active = constraints.get("is_active")
            if active and slot not in ACTIVE_SLOT_FILTERS:
                continue  # no outerwear for physical activities
            filters = [ACTIVE_SLOT_FILTERS[slot]] if active else []
            if constraints.get("is_messy") and slot in ("top", "bottom", "outerwear"):
                filters.append(MESSY_SLOT_FILTER)
            if is_smart(constraints) and slot == "bottom":
                filters.append(SMART_BOTTOM_FILTER)
            required = constraints.get("required", {})
            if "outerwear" in required and "top" in required and slot == "outerwear":
                pass  # handled by the filter below
            elif slot == "outerwear" and required and "outerwear" not in required:
                continue  # user named their pieces; don't add an unasked layer
            if slot in required:
                filters.append(f"subcategory = '{required[slot]}'")
            named = constraints.get("named", {})
            if slot in named:
                filters.append("original_filename IN ("
                               + ", ".join("'" + n.replace("'", "''") + "'" for n in named[slot]) + ")")
            elif constraints.get("required_color"):
                filters.append(f"color = '{constraints['required_color']}'")
            active_filter = psycopg.sql.SQL("".join(f" AND {f}" for f in filters))
            # Subcategories that can also fill this slot (e.g. hoodies for outerwear).
            extra_subcats = [sub for sub, slots in DUAL_SLOT_SUBCATEGORIES.items()
                             if slot in slots]

            # Tag filters first. Items with a missing (None) tag are let through.
            # Unavailable items (in the laundry) are never suggested.
            rows = conn.execute(
                psycopg.sql.SQL(
                    f"""SELECT {ITEM_COLUMNS} FROM {{}}
                        WHERE (category = %s OR subcategory = ANY(%s::text[]))
                          AND available
                          AND (formality IS NULL OR formality = ANY(%s)){{}}
                        ORDER BY clip_vec <=> %s::vector
                        LIMIT %s"""
                ).format(table, active_filter),
                (slot, extra_subcats, constraints["formality"],
                 query_vec, PER_CATEGORY),
            ).fetchall()
            fits = True

            # Small wardrobes often have nothing that passes the filters in some
            # category. Fall back to the closest items, flagged so GPT knows.
            if not rows:
                rows = conn.execute(
                    psycopg.sql.SQL(
                        f"""SELECT {ITEM_COLUMNS} FROM {{}}
                            WHERE (category = %s OR subcategory = ANY(%s::text[]))
                              AND available{{}}
                            ORDER BY clip_vec <=> %s::vector
                            LIMIT 3"""
                    ).format(table, active_filter),
                    (slot, extra_subcats, query_vec),
                ).fetchall()
                fits = False

            for row in rows:
                item = dict(zip(ITEM_COLUMNS.split(", "), row))
                item["id"] = str(item["id"])
                if item["id"] in seen_ids:  # already listed for another slot
                    continue
                seen_ids.add(item["id"])
                item["fits_filters"] = fits
                candidates.append(item)
    return candidates


# ---------------------------------------------------------------------------
# 3. Generate outfits with GPT.
# ---------------------------------------------------------------------------
def describe(short_id: str, item: dict) -> str:
    color = item.get("color") or "?"
    if color == "multicolor" and item.get("colors"):
        color = f"multicolor ({item['colors']})"
    slots = "/".join(s for s in SLOTS if s in eligible_slots(item))
    tags = " | ".join([slots, str(item.get("subcategory") or "?"),
                       f"{item.get('sleeve') or '?'} sleeve", f"{item.get('shade') or '?'} {color}",
                       str(item.get("pattern") or "?"), str(item.get("formality") or "?")])
    note = "" if item["fits_filters"] else "  (doesn't match the filters; use only if needed)"
    return f"{short_id}: {tags}  [{item.get('original_filename') or ''}]{note}"


# ---------------------------------------------------------------------------
# The feedback loop: your 👍 / 👎 (with an optional reason) and "I wore this" from the
# Today page, in the tables from add_feedback.sql. The rules:
#  - 👎 with "colours clash" / "don't like a piece": that outfit never comes back.
#  - any other 👎 (too formal, too casual, wrong occasion, no reason): it doesn't come
#    back for occasions at the same dress level, but can for others.
#  - 👍 and "I wore this": the AI leans towards them. A liked outfit may come back, at
#    most one per request, and not within a week of last being shown or worn.
#  - pieces worn in the last few days are avoided where possible.
# ---------------------------------------------------------------------------
EVERYWHERE_REASONS = {"colours clash", "don't like a piece"}   # about the outfit itself
REPEAT_GAP_DAYS = 7
RECENT_WEAR_DAYS = 3


def outfit_key(item_ids) -> str:
    """Same clothes = same outfit, whatever the slots or order."""
    return ",".join(sorted(str(i) for i in item_ids))


def occasion_of(formality) -> str:
    """The dress level a request was read as, e.g. ["smart casual", "casual"] -> "casual|smart casual"."""
    return "|".join(sorted(formality or []))


def load_feedback() -> dict:
    """Everything the engine has learned. Missing tables (add_feedback.sql not run) = nothing yet."""
    learned = {"votes": [], "worn_recently": {}, "recent_pieces": set()}
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            learned["votes"] = [
                {"key": k, "occasion": occ, "ids": [str(i) for i in ids], "vote": v, "reason": r,
                 "stale": stale}
                for k, occ, ids, v, r, stale in conn.execute(psycopg.sql.SQL(
                    """SELECT outfit_key, occasion, item_ids, vote, reason,
                              last_shown < now() - make_interval(days => %s)
                       FROM {} ORDER BY voted_at DESC LIMIT 200""").format(
                    psycopg.sql.Identifier(FEEDBACK_TABLE)), (REPEAT_GAP_DAYS,)).fetchall()]
            for k, ids, days_ago in conn.execute(psycopg.sql.SQL(
                    """SELECT outfit_key, item_ids, current_date - max(worn_on)
                       FROM {} WHERE worn_on > current_date - %s GROUP BY outfit_key, item_ids""").format(
                    psycopg.sql.Identifier(WEARS_TABLE)), (REPEAT_GAP_DAYS,)).fetchall():
                learned["worn_recently"][k] = days_ago
                if days_ago < RECENT_WEAR_DAYS:
                    learned["recent_pieces"].update(str(i) for i in ids)
    except psycopg.errors.UndefinedTable:
        pass
    return learned


def judge(learned: dict, key: str, occasion: str) -> str | None:
    """What the feedback says about one outfit for this request:
    "blocked", "liked" (may repeat), "liked-recently" (liked, but not again yet) or None."""
    liked = False
    for v in learned["votes"]:
        if v["key"] != key:
            continue
        if v["vote"] < 0 and (v["reason"] in EVERYWHERE_REASONS or v["occasion"] == occasion):
            return "blocked"
        if v["vote"] > 0:
            liked = liked or ("recent" if not v["stale"] else True)
    if key in learned["worn_recently"]:
        return "liked-recently"
    if liked == "recent":
        return "liked-recently"
    return "liked" if liked else None


def feedback_rules(learned: dict, items_by_short_id: dict, occasion: str) -> str:
    """Prompt lines describing past feedback with this request's short ids (A1, A2...).
    Only outfits whose clothes are all among today's candidates can be described."""
    short_of = {item["id"]: sid for sid, item in items_by_short_id.items()}
    describe_combo = lambda ids: " + ".join(sorted(short_of[i] for i in ids))
    liked, resting, disliked = [], [], []
    for v in learned["votes"]:
        if not all(i in short_of for i in v["ids"]):
            continue
        verdict = judge(learned, v["key"], occasion)
        combo = describe_combo(v["ids"])
        if verdict == "blocked":
            disliked.append(combo + (f" ({v['reason']})" if v["reason"] else ""))
        elif v["vote"] < 0 and v["reason"]:
            disliked.append(f"{combo} ({v['reason']} for another occasion; fine to use here if it suits)")
        elif verdict == "liked":
            liked.append(combo)
        elif verdict == "liked-recently":
            resting.append(combo)
    recent = sorted(short_of[i] for i in learned["recent_pieces"] if i in short_of)
    rules = ""
    if liked:
        rules += ("- The user LIKED these outfits: " + "; ".join(dict.fromkeys(liked[:8])) + ". "
                  "Lean towards similar pairings and style. You may repeat at most ONE of them; "
                  "the other outfits must be new.\n")
    if resting:
        rules += ("- The user also liked or recently wore these, but don't repeat them exactly right "
                  "now (similar style is welcome): " + "; ".join(dict.fromkeys(resting[:8])) + ".\n")
    if disliked:
        rules += ("- The user DISLIKED these outfits: " + "; ".join(dict.fromkeys(disliked[:8])) + ". "
                  "Never suggest the ones without a note, and avoid repeating what they disliked.\n")
    if recent:
        rules += ("- These pieces were worn in the last few days; avoid them if there's a good "
                  "alternative: " + ", ".join(recent) + ".\n")
    return rules


def generate_outfits(request: str, constraints: dict, items_by_short_id: dict,
                     n: int, learned: dict | None = None, already: list[str] | None = None) -> list[dict]:
    has_footwear = any(i["category"] == "footwear" for i in items_by_short_id.values())
    listing = "\n".join(describe(sid, item) for sid, item in items_by_short_id.items())

    prompt = (
        "You are a personal stylist for someone in Singapore.\n\n"
        f'Request: "{request}"\n'
        f"Target formality: {constraints['formality']}\n\n"
        "Wardrobe items (id: slots it can fill | subcategory | sleeve | color | pattern | formality):\n"
        f"{listing}\n\n"
        f"Build up to {n} different outfits using ONLY the ids above. Rules:\n"
        "- Each outfit is a top + bottom.\n"
        "- Every outfit must be complete: if a slot only has items marked "
        "\"doesn't match the filters\", still use the best of them rather than "
        "leaving the slot empty.\n"
        + ("- Each outfit must include footwear.\n" if has_footwear else "")
        + ("- The setting is cold: every outfit MUST be warm, meaning a long-sleeve "
           "top, or outerwear over a short-sleeve top.\n"
           if constraints.get("needs_warm_layer") else
           "- Leave out outerwear; the setting isn't cold.\n")
        + ("- This is a physical activity: use only short-sleeve tops, shorts or "
           "sweatpants, and sneakers (no sandals, slides, loafers or clogs), and no outerwear.\n"
           if constraints.get("is_active") else "")
        + ("- The user asked for these pieces; every outfit must use them: "
           + ", ".join(f"{sub} as the {slot}" for slot, sub in constraints["required"].items()) + ".\n"
           if constraints.get("required") else "")
        + ("- The user asked for these specific items of theirs; use one of them in "
           "that slot in every outfit: "
           + "; ".join(f"{slot}: {', '.join(ns)}" for slot, ns in constraints["named"].items()) + ".\n"
           if constraints.get("named") else "")
        + (f"- The whole outfit must be {constraints['required_color']}"
           + (" (except the specific items above)" if constraints.get("named") else "")
           + ".\n"
           if constraints.get("required_color") else "")
        + ("- This is a smart occasion: never use sweatpants or shorts.\n"
           if is_smart(constraints) else "")
        + ("- The food may splash or stain: avoid light-shade tops and bottoms; "
           "prefer dark shades.\n"
           if constraints.get("is_messy") else "")
        + "- Never put outerwear over a long-sleeve top; outerwear only goes over "
        "short-sleeve tops.\n"
        "- Put each id only in a slot it can fill (hoodies, sweatshirts and jackets "
        "may be the top or the outerwear, but not both in the same outfit).\n"
        "- Consider color harmony and avoid clashing patterns.\n"
        "- Make the outfits meaningfully different from each other.\n"
        + (feedback_rules(learned, items_by_short_id, occasion_of(constraints["formality"])) if learned else "")
        + ("- These outfits are already chosen; make yours different from them (other pieces "
           "where possible): " + "; ".join(already) + ".\n" if already else "")
        + "\n"
        'Reply with ONLY JSON: {"outfits": [{"top": id or null, "bottom": id or null, '
        '"outerwear": id or null, "footwear": id or null, '
        '"title": "short catchy outfit name", '
        '"reason": "one or two friendly sentences on why it works"}]}'
    )
    resp = llm.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content).get("outfits", [])


# ---------------------------------------------------------------------------
# 4. Validate: GPT can invent ids or put items in the wrong slot.
# ---------------------------------------------------------------------------
ACTIVE_BOTTOMS = ("shorts", "sweatpants")


def validate_outfits(raw_outfits: list, items_by_short_id: dict,
                     needs_warm_layer: bool = False, is_active: bool = False,
                     is_messy: bool = False, smart: bool = False,
                     required: dict | None = None,
                     required_color: str | None = None,
                     named: dict | None = None,
                     learned: dict | None = None,
                     occasion: str = "") -> list[dict]:
    has_footwear = any(i["category"] == "footwear" for i in items_by_short_id.values())
    valid, seen = [], set()
    repeats = 0

    for outfit in raw_outfits:
        if not isinstance(outfit, dict):
            continue
        chosen, ok = {}, True
        for slot in SLOTS:
            short_id = outfit.get(slot)
            if not short_id:
                continue
            # GPT sometimes drops the "A" prefix (8 instead of A8): accept both
            key = str(short_id).strip().upper()
            item = items_by_short_id.get(key) or items_by_short_id.get(f"A{key}")
            if item is None or slot not in eligible_slots(item):  # invented id / wrong slot
                ok = False
                break
            chosen[slot] = item
        if not ok:
            continue

        ids_used = [i["id"] for i in chosen.values()]
        if len(ids_used) != len(set(ids_used)):  # same item used in two slots
            continue

        # Two layering pieces (e.g. a sweatshirt worn as the top plus a jacket as
        # outerwear) is redundant stacking - at most one outerwear-eligible item.
        outerwear_eligible = sum(1 for i in chosen.values() if "outerwear" in eligible_slots(i))
        if outerwear_eligible > 1:
            continue
        # Outerwear only goes over short-sleeve tops, never over long sleeves.
        top_sleeve = chosen.get("top", {}).get("sleeve")
        if "outerwear" in chosen and top_sleeve == "long":
            continue
        # Cold settings need warmth: a long-sleeve top, or outerwear over a
        # short-sleeve one.
        if needs_warm_layer and top_sleeve != "long" and "outerwear" not in chosen:
            continue
        # Pieces the user asked for by name must be in the outfit.
        if required and any(chosen.get(slot, {}).get("subcategory") != sub
                            for slot, sub in required.items()):
            continue
        # Specific items the user named ("my air force") must fill their slot.
        named = named or {}
        if any(chosen.get(slot, {}).get("original_filename") not in ns for slot, ns in named.items()):
            continue
        # Whole-outfit colour the user asked for ("full white"): every piece
        # except the specific items they named.
        if required_color and any(i.get("color") != required_color
                                  for slot, i in chosen.items() if slot not in named):
            continue
        # Smart occasions: no sweatpants or shorts.
        if smart and chosen["bottom"].get("subcategory") in CASUAL_ONLY_BOTTOMS:
            continue
        # Messy food: no light-shade tops, bottoms or outerwear (stains show).
        if is_messy and any(chosen[s].get("shade") == "light"
                            for s in ("top", "bottom", "outerwear") if s in chosen):
            continue
        # Physical activity: short sleeves, sporty bottoms, casual non-sandal
        # footwear (loafers are smart casual), no outerwear.
        if is_active:
            shoe = chosen.get("footwear", {})
            if (top_sleeve != "short" or "outerwear" in chosen
                    or chosen["bottom"].get("subcategory") not in ACTIVE_BOTTOMS
                    or shoe.get("subcategory") == "sandals"
                    or (shoe and shoe.get("formality") != "casual")):
                continue

        has_base = "top" in chosen and "bottom" in chosen
        if not has_base or (has_footwear and "footwear" not in chosen):
            continue

        key = frozenset(i["id"] for i in chosen.values())
        if key in seen:  # duplicate outfit
            continue
        seen.add(key)
        # The feedback loop: blocked outfits never come back, liked ones not too often.
        verdict = judge(learned, outfit_key(key), occasion) if learned else None
        if verdict in ("blocked", "liked-recently"):
            continue
        if verdict == "liked":
            if repeats:      # at most one liked outfit per request
                continue
            repeats += 1

        valid.append({
            "items": {slot: {k: v for k, v in item.items() if k != "fits_filters"}
                      for slot, item in chosen.items()},
            "title": str(outfit.get("title", "")),
            "reason": str(outfit.get("reason", "")),
            "feedback": "up" if verdict == "liked" else None,   # you liked this exact outfit before
        })
    return valid


# ---------------------------------------------------------------------------
# The whole pipeline.
# ---------------------------------------------------------------------------
def recommend(request: str, n: int = 3) -> dict:
    constraints = parse_request(request)
    candidates = fetch_candidates(constraints)
    print(f"[OUTFIT] {request!r} -> {constraints}, {len(candidates)} candidates")

    # GPT is more reliable with short ids like A1 than with long UUIDs.
    items_by_short_id = {f"A{i + 1}": item for i, item in enumerate(candidates)}

    categories = {i["category"] for i in candidates}
    if not {"top", "bottom"} <= categories:
        return {"constraints": constraints, "outfits": [],
                "message": "Not enough items yet: tag at least one top and one bottom."}

    learned = load_feedback()
    occasion = occasion_of(constraints["formality"])
    if learned["votes"] or learned["worn_recently"]:
        ups = sum(1 for v in learned["votes"] if v["vote"] > 0)
        print(f"[FEEDBACK] {ups} liked, {len(learned['votes']) - ups} disliked, "
              f"{len(learned['worn_recently'])} worn this week")
    # Always aim for n outfits: ask GPT, keep the ones that pass every rule, ask again for
    # the missing ones (telling it what's already chosen), then fill any gap ourselves.
    short_of = {item["id"]: sid for sid, item in items_by_short_id.items()}
    combo = lambda o: " + ".join(f"{slot} {short_of[i['id']]}" for slot, i in o["items"].items())
    outfits, keys = [], set()

    def keep(found):
        for o in found:
            key = outfit_key(i["id"] for i in o["items"].values())
            liked_already = any(x.get("feedback") for x in outfits)
            if key in keys or (o.get("feedback") and liked_already):   # duplicate / 2nd liked repeat
                continue
            keys.add(key)
            outfits.append(o)

    for _attempt in range(3):  # first ask + up to two top-ups
        missing = n - len(outfits)
        if missing <= 0:
            break
        raw = generate_outfits(request, constraints, items_by_short_id, missing, learned,
                               already=[combo(o) for o in outfits])
        keep(validate_outfits(raw, items_by_short_id, constraints["needs_warm_layer"],
                              constraints["is_active"], constraints["is_messy"],
                              is_smart(constraints), constraints["required"],
                              constraints["required_color"], constraints["named"],
                              learned, occasion))
    if len(outfits) < n:
        filled = fill_outfits(items_by_short_id, constraints, learned, occasion, outfits, n - len(outfits))
        if filled:
            print(f"[OUTFIT] GPT gave {len(outfits)} usable outfits; filled {len(filled)} more from the rules")
        keep(filled)
    outfits = outfits[:n]
    mark_shown([outfit_key(i["id"] for i in o["items"].values()) for o in outfits if o.get("feedback")])

    result = {"constraints": constraints, "outfits": outfits}
    if not outfits:
        result["message"] = "Couldn't build a complete outfit from the current wardrobe."
    return result


# ---------------------------------------------------------------------------
# Filling the gap: when GPT's answers don't give n usable outfits, build the rest
# from the candidates with the same rules, preferring pieces not used yet.
# ---------------------------------------------------------------------------
def fill_outfits(items_by_short_id: dict, constraints: dict, learned: dict, occasion: str,
                 existing: list[dict], count: int) -> list[dict]:
    import itertools
    by_slot = {slot: [sid for sid, it in items_by_short_id.items() if slot in eligible_slots(it)]
               for slot in ("top", "bottom", "footwear", "outerwear")}
    used = {i["id"] for o in existing for i in o["items"].values()}
    taken = {outfit_key(i["id"] for i in o["items"].values()) for o in existing}
    # Outerwear only when it's cold (then it may be needed over a short-sleeve top)
    layers = [None] + (by_slot["outerwear"] if constraints.get("needs_warm_layer") else [])
    combos = []
    for top, bottom, shoe, layer in itertools.islice(itertools.product(
            by_slot["top"], by_slot["bottom"], by_slot["footwear"] or [None], layers), 4000):
        raw = {"top": top, "bottom": bottom, "footwear": shoe, "outerwear": layer}
        ids = [items_by_short_id[sid]["id"] for sid in raw.values() if sid]
        fits = sum(1 for sid in raw.values() if sid and items_by_short_id[sid].get("fits_filters"))
        combos.append((raw, ids, fits))

    def check(raw):
        return validate_outfits([raw], items_by_short_id, constraints["needs_warm_layer"],
                                constraints["is_active"], constraints["is_messy"], is_smart(constraints),
                                constraints["required"], constraints["required_color"], constraints["named"],
                                learned, occasion)

    filled = []
    while len(filled) < count:
        # Best next: most pieces not used in the other outfits, then most that fit the filters
        ranked = sorted((c for c in combos if outfit_key(c[1]) not in taken),
                        key=lambda c: (sum(1 for i in c[1] if i not in used), c[2]), reverse=True)
        pick = None
        for raw, ids, _ in ranked[:300]:
            valid = check(raw)
            taken.add(outfit_key(ids))
            if valid:
                pick = valid[0]
                break
        if pick is None:
            break   # the wardrobe can't make another valid outfit for this request
        names = [itemName_py(i) for i in pick["items"].values()]
        pick["title"] = ("Another option", "One more idea", "Easy pick", "Back-up look")[len(filled) % 4]
        pick["reason"] = "A simple mix: " + ", ".join(names) + "."
        used.update(i["id"] for i in pick["items"].values())
        filled.append(pick)
    return filled


def itemName_py(item: dict) -> str:
    """'macos_43_blue_striped_polo.png' -> 'blue striped polo' (same idea as the website)."""
    import re
    name = re.sub(r"\.[a-z0-9]+$", "", str(item.get("original_filename") or item.get("subcategory") or "item"), flags=re.I)
    name = re.sub(r"^macos_\d+_", "", name)
    return name.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Swap one piece: keep the rest of the outfit, offer alternatives for one slot.
# ---------------------------------------------------------------------------
SLOT_WORDS = {"top": "top", "bottom": "bottom", "outerwear": "layer", "footwear": "shoes"}


def swap_options(request: str, constraints: dict, current: dict, slot: str, n: int = 3) -> list[dict]:
    """current = {slot: item id} for the outfit on screen. Returns up to n alternatives
    for `slot`, each a complete outfit that passes the same rules (and feedback) as
    normal suggestions: [{"items": {...}, "item": the new piece, "reason": "..."}]."""
    candidates = fetch_candidates(constraints)
    items_by_short_id = {f"A{i + 1}": item for i, item in enumerate(candidates)}
    short_of = {item["id"]: sid for sid, item in items_by_short_id.items()}
    kept = {s: short_of[i] for s, i in current.items() if s != slot and i in short_of}
    if len(kept) < len([s for s in current if s != slot]):
        return []   # a kept piece is no longer available (e.g. in the laundry)
    options = [sid for sid, item in items_by_short_id.items()
               if slot in eligible_slots(item) and item["id"] != current.get(slot) and sid not in kept.values()]
    if not options:
        return []
    learned = load_feedback()
    occasion = occasion_of(constraints["formality"])

    def check(sid):   # the outfit with this piece swapped in, if it passes every rule
        raw = {**kept, slot: sid}
        return validate_outfits([raw], items_by_short_id, constraints["needs_warm_layer"],
                                constraints["is_active"], constraints["is_messy"], is_smart(constraints),
                                constraints["required"], constraints["required_color"], constraints["named"],
                                learned, occasion)

    listing = "\n".join(describe(sid, items_by_short_id[sid]) for sid in options)
    outfit_now = ", ".join(f"{s}: {describe(sid, items_by_short_id[sid])}" for s, sid in kept.items())
    prompt = (
        "You are a personal stylist for someone in Singapore.\n\n"
        f'Request: "{request}"\nTarget formality: {constraints["formality"]}\n\n'
        f"They're keeping these pieces: {outfit_now}.\n"
        f"They want a different {SLOT_WORDS[slot]}. Choose the {n + 2} best replacements from:\n{listing}\n\n"
        "Pick ones that suit the request and go well with the kept pieces (colour harmony, no clashing "
        "patterns, matching formality), and make them different from each other.\n"
        + feedback_rules(learned, items_by_short_id, occasion)
        + '\nReply with ONLY JSON: {"options": [{"id": id, "reason": "one short friendly sentence"}]}'
    )
    resp = llm.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=400, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    picked = json.loads(resp.choices[0].message.content).get("options", [])

    results, used = [], set()
    for opt in picked:
        sid = str(opt.get("id", "")).strip().upper()
        sid = sid if sid in items_by_short_id else f"A{sid}"
        if sid in used or sid not in options:
            continue
        valid = check(sid)
        if valid:
            used.add(sid)
            results.append({"items": valid[0]["items"], "item": valid[0]["items"][slot],
                            "reason": str(opt.get("reason", ""))})
        if len(results) == n:
            break
    return results


# ---------------------------------------------------------------------------
# Readable terminal output.
# ---------------------------------------------------------------------------
def format_text(request: str, result: dict) -> str:
    c = result["constraints"]
    lines = [f"Request: {request}",
             f"Looking for: {' / '.join(c['formality'])}", ""]
    for n, outfit in enumerate(result["outfits"], 1):
        lines.append(f"Outfit {n}: {outfit.get('title') or ''}".rstrip(": "))
        for slot, item in outfit["items"].items():
            kind = item.get("subcategory") or item["category"]
            lines.append(f"  {slot.capitalize():<10}{item['color']} {kind}"
                         f"  ({os.path.basename(item['image_path'])})")
        lines += [f"  Why: {outfit['reason']}", ""]
    if result.get("message"):
        lines.append(result["message"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML preview: outfits.html shows each outfit as a card with its photos side by
# side, so you can judge by eye whether the pieces look good together.
# ---------------------------------------------------------------------------
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outfits.html")


def render_html(request: str, result: dict) -> str:
    cards = []
    for n, outfit in enumerate(result["outfits"], 1):
        photos = "".join(
            f'<figure><img src="{quote(item["image_path"])}" alt="{esc(item["original_filename"])}">'
            f'<figcaption>{esc(slot)}: {esc(str(item["color"]))} '
            f'{esc(str(item.get("subcategory") or item["category"]))}</figcaption></figure>'
            for slot, item in outfit["items"].items()
        )
        title = esc(outfit.get("title") or f"Outfit {n}")
        cards.append(f'<section class="card"><h2>{title}</h2><div class="photos">{photos}</div>'
                     f'<p>{esc(outfit["reason"])}</p></section>')
    if result.get("message"):
        cards.append(f"<p>{esc(result['message'])}</p>")
    body = "".join(cards)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Outfits</title><style>
body{{font-family:system-ui,sans-serif;background:#f4f2ee;margin:0;padding:24px;color:#222}}
h1{{font-size:20px;font-weight:600}} .card{{background:#fff;border-radius:16px;padding:20px;margin:0 0 24px;box-shadow:0 2px 8px #0001}}
.card h2{{margin:0 0 12px;font-size:18px}} .photos{{display:flex;gap:12px;flex-wrap:wrap}}
figure{{margin:0;flex:1 1 140px;max-width:240px;text-align:center}}
img{{width:100%;aspect-ratio:2/3;object-fit:contain;background:#faf9f7;border-radius:12px}}
figcaption{{font-size:12px;color:#777;margin-top:4px}} p{{margin:12px 0 0;line-height:1.4}}
</style></head><body><h1>{esc(request)}</h1>{body}</body></html>"""


def write_preview(request: str, result: dict, open_browser: bool = False) -> str:
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(render_html(request, result))
    if open_browser:
        webbrowser.open("file://" + HTML_PATH)
    return HTML_PATH


# ---------------------------------------------------------------------------
# Endpoint, added to the same FastAPI app as /items.
# ---------------------------------------------------------------------------
class OutfitRequest(BaseModel):
    request: str
    n: int = 3


@app.post("/outfits")
def outfits_endpoint(body: OutfitRequest):
    result = recommend(body.request, max(1, min(body.n, 5)))
    write_preview(body.request, result)  # refreshes outfits.html, doesn't open it
    return result


def mark_shown(keys: list[str]) -> None:
    """A liked outfit was suggested again: start its one-week rest."""
    if not keys:
        return
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            conn.execute(psycopg.sql.SQL("UPDATE {} SET last_shown = now() WHERE outfit_key = ANY(%s) AND vote = 1")
                         .format(psycopg.sql.Identifier(FEEDBACK_TABLE)), (keys,))
    except psycopg.errors.UndefinedTable:
        pass


def need_tables(run):
    try:
        return run()
    except psycopg.errors.UndefinedTable:
        raise HTTPException(503, "The feedback tables don't exist yet: run add_feedback.sql")


DOWN_REASONS = ("colours clash", "don't like a piece", "too formal", "too casual", "not for this occasion")


class SwapRequest(BaseModel):
    request: str
    items: dict[str, str]            # the outfit on screen: {slot: item id}
    slot: str                        # which piece to swap
    constraints: dict | None = None  # result.constraints from /outfits (skips re-reading the request)
    n: int = 3


@app.post("/outfits/swap")
def swap_endpoint(body: SwapRequest):
    """"Try different shoes": up to n alternatives for one piece, keeping the rest."""
    if body.slot not in SLOTS:
        raise HTTPException(422, f"slot must be one of {SLOTS}")
    constraints = body.constraints or parse_request(body.request)
    for key, default in (("required", {}), ("named", {}), ("required_color", None), ("needs_warm_layer", False),
                         ("is_active", False), ("is_messy", False), ("formality", ["casual"])):
        constraints.setdefault(key, default)
    options = swap_options(body.request, constraints, body.items, body.slot, max(1, min(body.n, 5)))
    print(f"[SWAP] {body.slot} for {body.request!r}: {len(options)} options")
    return {"slot": body.slot, "options": options}


class FeedbackVote(BaseModel):
    item_ids: list[str]              # the clothes in the outfit
    vote: str | None                 # "up", "down", or null to take the vote back
    formality: list[str] = []        # how the request was read (result.constraints.formality)
    reason: str | None = None        # for "down": one of DOWN_REASONS (optional)
    request: str | None = None
    title: str | None = None


@app.post("/feedback")
def save_feedback(body: FeedbackVote):
    """👍 / 👎 on an outfit, for this occasion. Voting again on the same outfit and occasion
    replaces the vote; vote null takes it back."""
    if not body.item_ids:
        raise HTTPException(400, "item_ids is empty")
    if body.vote not in ("up", "down", None):
        raise HTTPException(422, 'vote must be "up", "down" or null')
    reason = (body.reason or "").lower().strip() or None
    if reason and (body.vote != "down" or reason not in DOWN_REASONS):
        raise HTTPException(422, f"reason must be one of {list(DOWN_REASONS)}, and only with a down vote")
    key, occasion = outfit_key(body.item_ids), occasion_of(body.formality)
    table = psycopg.sql.Identifier(FEEDBACK_TABLE)

    def run():
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            if body.vote is None:
                conn.execute(psycopg.sql.SQL("DELETE FROM {} WHERE outfit_key = %s AND occasion = %s")
                             .format(table), (key, occasion))
            else:
                conn.execute(psycopg.sql.SQL(
                    """INSERT INTO {} (outfit_key, occasion, item_ids, vote, reason, request, title)
                       VALUES (%s, %s, %s::uuid[], %s, %s, %s, %s)
                       ON CONFLICT (outfit_key, occasion) DO UPDATE
                       SET vote = EXCLUDED.vote, reason = EXCLUDED.reason, request = EXCLUDED.request,
                           title = EXCLUDED.title, voted_at = now(), last_shown = now()""").format(table),
                    (key, occasion, sorted(body.item_ids), 1 if body.vote == "up" else -1,
                     reason, body.request, body.title))
    need_tables(run)
    print(f"[FEEDBACK] {body.vote or 'cleared'}{f' ({reason})' if reason else ''} for {occasion}: {body.title!r}")
    return {"outfit_key": key, "occasion": occasion, "vote": body.vote, "reason": reason}


class WearBody(BaseModel):
    item_ids: list[str]
    worn: bool = True                # false = undo an "I wore this"
    worn_on: str | None = None       # which day to undo (YYYY-MM-DD); default today
    formality: list[str] = []
    request: str | None = None
    title: str | None = None


@app.post("/wears")
def save_wear(body: WearBody):
    """"I wore this" (today). Also counts as a 👍 for this occasion."""
    if not body.item_ids:
        raise HTTPException(400, "item_ids is empty")
    key = outfit_key(body.item_ids)

    def run():
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            if not body.worn:
                conn.execute(psycopg.sql.SQL(
                    "DELETE FROM {} WHERE outfit_key = %s AND worn_on = COALESCE(%s::date, current_date)")
                    .format(psycopg.sql.Identifier(WEARS_TABLE)), (key, body.worn_on))
                return
            conn.execute(psycopg.sql.SQL(
                """INSERT INTO {} (outfit_key, item_ids, request, title) VALUES (%s, %s::uuid[], %s, %s)
                   ON CONFLICT (outfit_key, worn_on) DO NOTHING""").format(
                psycopg.sql.Identifier(WEARS_TABLE)), (key, sorted(body.item_ids), body.request, body.title))
    need_tables(run)
    if body.worn:
        save_feedback(FeedbackVote(item_ids=body.item_ids, vote="up", formality=body.formality,
                                   request=body.request, title=body.title))
    print(f"[WORE] {'yes' if body.worn else 'undone'}: {body.title!r}")
    return {"outfit_key": key, "worn": body.worn}


@app.get("/history")
def history():
    """Everything for the History page: every vote and every wear, newest first,
    with the clothes' pictures and names (including items now in the laundry)."""
    def run():
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            votes = conn.execute(psycopg.sql.SQL(
                """SELECT outfit_key, occasion, item_ids, vote, reason, request, title, voted_at
                   FROM {} ORDER BY voted_at DESC""").format(psycopg.sql.Identifier(FEEDBACK_TABLE))).fetchall()
            wears = conn.execute(psycopg.sql.SQL(
                """SELECT outfit_key, worn_on, item_ids, request, title
                   FROM {} ORDER BY worn_on DESC""").format(psycopg.sql.Identifier(WEARS_TABLE))).fetchall()
            ids = {str(i) for row in votes for i in row[2]} | {str(i) for row in wears for i in row[2]}
            items = {}
            if ids:
                for iid, path, name, cat in conn.execute(psycopg.sql.SQL(
                        "SELECT id, image_path, original_filename, category FROM {} WHERE id = ANY(%s::uuid[])")
                        .format(psycopg.sql.Identifier(ITEMS_TABLE)), (list(ids),)).fetchall():
                    items[str(iid)] = {"id": str(iid), "image_path": path, "original_filename": name, "category": cat}
        return votes, wears, items
    votes, wears, items = need_tables(run)
    pieces = lambda ids: [items[str(i)] for i in ids if str(i) in items]   # removed items drop out
    return {
        "votes": [{"outfit_key": k, "occasion": occ, "formality": occ.split("|") if occ else [],
                   "items": pieces(ids), "item_ids": [str(i) for i in ids],
                   "vote": "up" if v > 0 else "down", "reason": r, "request": req, "title": t,
                   "when": at.isoformat()} for k, occ, ids, v, r, req, t, at in votes],
        "wears": [{"outfit_key": k, "worn_on": d.isoformat(), "items": pieces(ids),
                   "item_ids": [str(i) for i in ids], "request": req, "title": t}
                  for k, d, ids, req, t in wears],
    }


@app.get("/feedback")
def list_feedback():
    """What the engine has learned (to check it): votes newest first, and this week's wears."""
    learned = load_feedback()
    return {"votes": [{**v, "vote": "up" if v["vote"] > 0 else "down"} for v in learned["votes"]],
            "worn_this_week": learned["worn_recently"]}


# ---------------------------------------------------------------------------
# The website: http://127.0.0.1:8000/ opens the app (the pages live in web/).
# Item photos are served from the folders their image_path points into.
# ---------------------------------------------------------------------------
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class FreshStaticFiles(StaticFiles):
    """Tell the browser to check for a newer copy every time, so after a page,
    style.css or common.js changes you see it on a normal refresh (no stale mix of old and new)."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/app", FreshStaticFiles(directory=os.path.join(PROJECT_DIR, "web"), html=True), name="web")
app.mount("/macos_generate", StaticFiles(directory=os.path.join(PROJECT_DIR, "macos_generate")),
          name="macos_generate")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")  # same folder POST /items saves to


# Small versions of clothing photos for grids and thumbnails (the full photos are ~1 MB each).
# Made the first time they're asked for, then kept in thumbs/ (git-ignored, safe to delete).
THUMB_DIR = os.path.join(PROJECT_DIR, "thumbs")
THUMB_SIZE = 400                                   # longest side, pixels
THUMB_ROOTS = ("macos_generate", "uploads")        # only photos from these folders


@app.get("/thumb/{path:path}", include_in_schema=False)
def thumbnail(path: str):
    from fastapi.responses import FileResponse
    from PIL import Image as PILImage
    src = os.path.normpath(os.path.join(PROJECT_DIR, path))
    if not any(src.startswith(os.path.join(PROJECT_DIR, root) + os.sep) for root in THUMB_ROOTS) \
            or not os.path.isfile(src):
        raise HTTPException(404, "No such photo")
    out = os.path.join(THUMB_DIR, path.replace("/", "__").rsplit(".", 1)[0] + ".webp")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        os.makedirs(THUMB_DIR, exist_ok=True)
        with PILImage.open(src) as img:
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(out, "WEBP", quality=82, method=4)   # keeps the transparent background
    return FileResponse(out, media_type="image/webp", headers={"Cache-Control": "max-age=86400"})


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/app/")


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "--json"]  # --json prints raw JSON instead
    text = " ".join(args) or "casual outfit for lunch with friends"
    result = recommend(text)
    if "--json" in sys.argv:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_text(text, result))
    print(f"[PREVIEW] {write_preview(text, result, open_browser=True)}")
