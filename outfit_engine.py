"""
Outfit engine for the wardrobe app.

Turns a request like "dinner date tonight, we'll walk around outside" into
2-3 outfits built only from items already tagged by tagging_worker_2.py.

  1. Parse   - GPT turns the request into tag filters + a style description
  2. Retrieve - Postgres filters by tags, then CLIP ranks items by similarity
                to the style description (a shortlist per category)
  3. Generate - GPT picks items from the shortlist and builds outfits
  4. Validate - code checks every item exists and each outfit is complete

It reuses ALLOWED and CLIP from tagging_worker_2.py so the tag lists live in
one place, but calls GPT (OpenAI) for the parsing/generation steps.
Put this file next to tagging_worker_2.py.

Reads items from the view_items_2 table (see view_items_2.sql).

Env vars:
  OPENAI_API_KEY   - for the LLM
  OPENAI_MODEL     - optional, any GPT model (default below)

Run the server (serves BOTH /items and /outfits):
  uvicorn outfit_engine:app --reload

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
from pydantic import BaseModel

from tagging_worker_2 import ALLOWED, _as_tensor, app, clip_model, clip_proc

llm = OpenAI()  # reads OPENAI_API_KEY
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Which table/view to read items from.
ITEMS_TABLE = os.environ.get("ITEMS_TABLE", "view_items_2")

# How many candidates to shortlist per category before GPT chooses.
PER_CATEGORY = 6

# Outfit slots. An outfit is (top + bottom) or a dress, plus shoes,
# with outerwear as an optional layer.
SLOTS = ["top", "bottom", "dress", "outerwear", "shoes"]


# ---------------------------------------------------------------------------
# 1. Parse the request into filters.
# ---------------------------------------------------------------------------
def parse_request(request: str) -> dict:
    prompt = (
        "A user in Singapore wants an outfit. Their request:\n"
        f'"{request}"\n\n'
        "Reply with ONLY a JSON object with these keys:\n"
        '- "formality": list of acceptable values from '
        f"{ALLOWED['formality']}\n"
        '- "search_text": one short English sentence describing the ideal outfit\'s '
        'look, e.g. "a relaxed smart-casual outfit for a warm evening"\n\n'
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
    return constraints


# ---------------------------------------------------------------------------
# 2. Retrieve a shortlist per category.
# ---------------------------------------------------------------------------
def clip_text_vector(text: str) -> np.ndarray:
    """Put the style description on the same 'meaning map' as the item photos."""
    with torch.no_grad():
        inputs = clip_proc(text=[text], return_tensors="pt", padding=True)
        vec = _as_tensor(clip_model.get_text_features(**inputs))
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].numpy().astype(np.float32)


ITEM_COLUMNS = "id, image_path, original_filename, category, color, pattern, formality"


def fetch_candidates(constraints: dict) -> list[dict]:
    query_vec = clip_text_vector(constraints["search_text"])
    candidates = []
    table = psycopg.sql.Identifier(ITEMS_TABLE)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        for category in ALLOWED["category"]:
            # Tag filters first. Items with a missing (None) tag are let through.
            rows = conn.execute(
                psycopg.sql.SQL(
                    f"""SELECT {ITEM_COLUMNS} FROM {{}}
                        WHERE category = %s
                          AND (formality IS NULL OR formality = ANY(%s))
                        ORDER BY clip_vec <=> %s::vector
                        LIMIT %s"""
                ).format(table),
                (category, constraints["formality"],
                 query_vec, PER_CATEGORY),
            ).fetchall()
            fits = True

            # Small wardrobes often have nothing that passes the filters in some
            # category. Fall back to the closest items, flagged so GPT knows.
            if not rows:
                rows = conn.execute(
                    psycopg.sql.SQL(
                        f"""SELECT {ITEM_COLUMNS} FROM {{}}
                            WHERE category = %s
                            ORDER BY clip_vec <=> %s::vector
                            LIMIT 3"""
                    ).format(table),
                    (category, query_vec),
                ).fetchall()
                fits = False

            for row in rows:
                item = dict(zip(ITEM_COLUMNS.split(", "), row))
                item["id"] = str(item["id"])
                item["fits_filters"] = fits
                candidates.append(item)
    return candidates


# ---------------------------------------------------------------------------
# 3. Generate outfits with GPT.
# ---------------------------------------------------------------------------
def describe(short_id: str, item: dict) -> str:
    tags = " | ".join(str(item.get(k) or "?") for k in
                      ("category", "color", "pattern", "formality"))
    note = "" if item["fits_filters"] else "  (doesn't match the filters; use only if needed)"
    return f"{short_id}: {tags}  [{item.get('original_filename') or ''}]{note}"


def generate_outfits(request: str, constraints: dict, items_by_short_id: dict,
                     n: int) -> list[dict]:
    has_shoes = any(i["category"] == "shoes" for i in items_by_short_id.values())
    listing = "\n".join(describe(sid, item) for sid, item in items_by_short_id.items())

    prompt = (
        "You are a personal stylist for someone in Singapore (hot and humid "
        "outdoors, cold air-conditioning indoors).\n\n"
        f'Request: "{request}"\n'
        f"Target formality: {constraints['formality']}\n\n"
        "Wardrobe items (id: category | color | pattern | formality):\n"
        f"{listing}\n\n"
        f"Build up to {n} different outfits using ONLY the ids above. Rules:\n"
        "- Each outfit is either a top + bottom, or a dress.\n"
        + ("- Each outfit must include shoes.\n" if has_shoes else "")
        + "- Add outerwear only when it genuinely helps (e.g. air-conditioned venues).\n"
        "- Put each id only in the slot matching its category.\n"
        "- Consider color harmony and avoid clashing patterns.\n"
        "- Make the outfits meaningfully different from each other.\n\n"
        'Reply with ONLY JSON: {"outfits": [{"top": id or null, "bottom": id or null, '
        '"dress": id or null, "outerwear": id or null, "shoes": id or null, '
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
def validate_outfits(raw_outfits: list, items_by_short_id: dict) -> list[dict]:
    has_shoes = any(i["category"] == "shoes" for i in items_by_short_id.values())
    valid, seen = [], set()

    for outfit in raw_outfits:
        if not isinstance(outfit, dict):
            continue
        chosen, ok = {}, True
        for slot in SLOTS:
            short_id = outfit.get(slot)
            if not short_id:
                continue
            item = items_by_short_id.get(str(short_id).strip())
            if item is None or item["category"] != slot:  # invented id / wrong slot
                ok = False
                break
            chosen[slot] = item
        if not ok:
            continue

        has_base = ("top" in chosen and "bottom" in chosen) or "dress" in chosen
        if not has_base or (has_shoes and "shoes" not in chosen):
            continue
        if "dress" in chosen and ("top" in chosen or "bottom" in chosen):
            continue

        key = frozenset(i["id"] for i in chosen.values())
        if key in seen:  # duplicate outfit
            continue
        seen.add(key)

        valid.append({
            "items": {slot: {k: v for k, v in item.items() if k != "fits_filters"}
                      for slot, item in chosen.items()},
            "title": str(outfit.get("title", "")),
            "reason": str(outfit.get("reason", "")),
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
    if not ({"top", "bottom"} <= categories or "dress" in categories):
        return {"constraints": constraints, "outfits": [],
                "message": "Not enough items yet: tag at least one top and one "
                           "bottom, or a dress."}

    outfits = []
    for _attempt in range(2):  # one retry if GPT's first answer is unusable
        raw = generate_outfits(request, constraints, items_by_short_id, n)
        outfits = validate_outfits(raw, items_by_short_id)[:n]
        if outfits:
            break

    result = {"constraints": constraints, "outfits": outfits}
    if not outfits:
        result["message"] = "Couldn't build a complete outfit from the current wardrobe."
    return result


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
            lines.append(f"  {slot.capitalize():<10}{item['color']} {item['category']}"
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
            f'<figcaption>{esc(slot)}: {esc(str(item["color"]))} {esc(str(item["category"]))}</figcaption></figure>'
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
