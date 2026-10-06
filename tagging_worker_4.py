"""
Tagging worker for a wardrobe app (version 4: GPT tags, CLIP vectors).

Runs once per uploaded clothing photo:
  1. Tags it with OpenAI GPT-5 mini (category, subcategory, color, pattern, formality)
  2. Computes its FashionCLIP image vector (used later by the outfit engine for search)
  3. Saves both to the database

What's new in version 4:
  - Tags come from OpenAI (cloud API) instead of CLIP zero-shot / Ollama.
    The model's answer is locked to the allowed values with a JSON schema.
  - CLIP is only used for the search vector, so CLIP_PHRASES and the
    confidence thresholds are gone.
  - If OpenAI fails, the item is still saved with its vector and empty tags
    (None = needs review), instead of being lost.

Install:
  pip install fastapi uvicorn python-multipart torch transformers pillow \
              psycopg[binary] pgvector python-dotenv openai

Env vars:
  DATABASE_URL        - Postgres with the pgvector extension
  OPENAI_API_KEY      - OpenAI API key
  OPENAI_MODEL        - optional, default "gpt-5-mini"
  ITEMS_TABLE         - optional, table to write to (default "items")

Run:
  uvicorn tagging_worker_4:app --reload
"""

import base64
import contextlib
import functools
import io
import json
import os
import traceback
import uuid

import psycopg
import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, UploadFile
from openai import OpenAI
from pgvector.psycopg import register_vector
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()  # uvicorn doesn't read .env on its own

# Which table to save to (same schema as items). Lets a one-off batch run
# write to a separate table without touching the live upload endpoint.
ITEMS_TABLE = os.environ.get("ITEMS_TABLE", "items")

# ---------------------------------------------------------------------------
# 1. The allowed tags. The model can only pick from these lists.
# ---------------------------------------------------------------------------
ALLOWED = {
    "category":    ["top", "bottom", "outerwear", "footwear"],
    "subcategory": ["t-shirts", "polos", "shirts", "hoodies", "sweatshirts", "jackets",
                    "shorts", "jeans", "sweatpants",
                    "coats", "shoes"],
    "color":       ["black", "white", "grey", "blue", "beige", "brown",
                    "green", "red", "orange", "pink", "yellow", "purple",
                    "multicolor"],
    "pattern":     ["solid", "striped", "checked", "floral", "graphic", "camo"],
    "formality":   ["casual", "smart casual", "business casual", "formal"],
}

# Only the borderline shades that tend to get mislabelled.
COLOR_GUIDE = {
    "black":  "incl. faded/washed black, black denim; faded denim is black, even with a cool tint",
    "grey":   "incl. charcoal, dark slate; mottled, speckled or heathered fabric or any dull cool shade is grey, even with a cool tint",
    "white":  "incl. off-white, cream, ivory, chalk; any very light shade with only a faint warm tint is white",
    "blue":   "incl. navy, indigo, mid-blue, light blue; only clearly saturated shades, never dull or muted ones",
    "beige":  "only pale sand or cream-tan; never mid-tone",
    "brown":  "incl. tan, camel, taupe, khaki, chocolate; any mid-tone tan or taupe is brown, even when muted",
    "green":  "incl. olive, teal, forest, bottle; deep or dark shades are still green",
    "red":    "incl. crimson, raspberry, burgundy; any saturated or deep shade is red",
    "pink":   "only pale pink; never saturated or deep",
    "purple": "incl. plum",
}

# Instructions for the model.
PROMPT = (
    "Classify this clothing item. Each value must be one of:\n"
    + json.dumps(ALLOWED)
    + "\nColour notes: " + "; ".join(f"{k} {v}" for k, v in COLOR_GUIDE.items())
    + ".\nIn colors, list every fabric colour that stands out with its rough % "
    "of the item (adding up to 100), including thin stripes and trims. Keep light "
    "and dark shades of one colour as separate entries. Ignore logos and printed graphics.\n"
)

# The colours a multicolor item can be broken into (every color but multicolor).
MULTI_COLORS = [c for c in ALLOWED["color"] if c != "multicolor"]

# The model doesn't pick color itself: it lists the colours that stand out
# with their shares, and color_from_shares() decides the color from those.
MAIN_MIN = 60      # % one colour must cover for the item to be named after it
STANDOUT_MIN = 5   # % from which a listed colour counts as standing out
MULTI_COUNT = 3    # this many standout colours (shades separate) = multicolor
PANEL_SHARE = 25   # a colour this big counts as a panel, whatever the model calls it
PANEL_COUNT = 2    # ...even with a MAIN_MIN colour, this many other panel colours = multicolor
ACCENT_MIN = 7     # ...or this many other colours of at least this % (any kind)
PANEL_MIN = 6      # a "panel" smaller than this is just a trace and doesn't count

# JSON schema built from ALLOWED, so the model can only answer with allowed values.
TAG_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "string", "enum": v} for k, v in ALLOWED.items() if k != "color"},
        "colors": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": MULTI_COLORS},
                "share": {"type": "integer",
                          "description": "Rough % of the item this colour covers"},
                "kind": {"type": "string", "enum": ["panel", "stripe/trim"],
                         "description": "'panel' for whole sections, large areas, wide bands or colour-blocking, "
                                        "'stripe/trim' for thin stripes, piping, trims, edging and small details "
                                        "(collars, cuffs, soles, linings)"},
            },
            "required": ["color", "share", "kind"],
            "additionalProperties": False,
        }},
    },
    "required": [*[k for k in ALLOWED if k != "color"], "colors"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# 2. Load CLIP once when the server starts (not on every upload).
#    Must match tagging_worker_2.py's CLIP_NAME: the outfit engines embed
#    their search text with that file's model and compare it against the
#    vectors stored here.
# ---------------------------------------------------------------------------
CLIP_NAME = "patrickjohncyh/fashion-clip"


@functools.cache
def load_clip():
    """Loaded on first use, so scripts that only tag don't pay for it."""
    return CLIPModel.from_pretrained(CLIP_NAME).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)


def _as_tensor(out):
    """Newer transformers versions return an output object instead of a tensor."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def clip_image_vector(img: Image.Image) -> list[float]:
    """The item's location on CLIP's 'meaning map' (512 numbers, normalized)."""
    with torch.no_grad():
        clip_model, clip_proc = load_clip()
        inputs = clip_proc(images=img, return_tensors="pt")
        vec = _as_tensor(clip_model.get_image_features(**inputs))
        return (vec / vec.norm(dim=-1, keepdim=True))[0].tolist()


# ---------------------------------------------------------------------------
# 3. Tagging: OpenAI (GPT-5 mini by default).
# ---------------------------------------------------------------------------
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
llm = OpenAI()  # reads OPENAI_API_KEY


MAX_SIDE = 512  # longest side sent to the model (faster, cheaper)


def shrink(image_bytes: bytes) -> bytes:
    """Resize so the longest side is MAX_SIDE; colour doesn't need full resolution.

    Transparency is kept as-is: tested against a grey backdrop, GPT-5 mini
    scored the same or better on colour with the transparent cut-outs.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tag_with_openai(image_bytes: bytes, media_type: str) -> dict:
    image_bytes, media_type = shrink(image_bytes), "image/png"
    data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"
    resp = llm.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": PROMPT},
        ]}],
        # strict: output must match the allowed values exactly
        response_format={"type": "json_schema", "json_schema": {
            "name": "tags", "schema": TAG_SCHEMA, "strict": True,
        }},
        reasoning_effort="low",  # GPT-5 models don't take temperature
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# 4. Validate: never trust model output blindly.
# ---------------------------------------------------------------------------
def validate(tags: dict) -> dict:
    clean = {}
    for attribute, options in ALLOWED.items():
        value = str(tags.get(attribute, "")).lower().strip()
        clean[attribute] = value if value in options else None  # None = needs review
    if clean["category"] == "footwear" or clean["subcategory"] == "shoes":
        clean["category"], clean["subcategory"] = "footwear", "shoes"
    clean["color"], clean["colors"] = color_from_shares(tags.get("colors"), clean["pattern"])
    # Camo is always multicolor, even when the model lists its shades as one colour
    if clean["pattern"] == "camo" and clean["color"] is not None:
        names = [str(c.get("color")).lower() for c in tags.get("colors") or [] if isinstance(c, dict)]
        clean["color"] = "multicolor"
        clean["colors"] = list(dict.fromkeys(n for n in names if n in MULTI_COLORS))[:4] or None
    return clean


def color_from_shares(raw, pattern=None) -> tuple[str | None, list[str] | None]:
    """Decide (color, colors) from the model's colour shares.

    1. One colour covering MAIN_MIN % or more names the item, whatever its
       stripes or trims (black shorts with blue and white side stripes are
       black), unless PANEL_COUNT or more other colours are whole panels
       (white sneakers with pink and yellow panels are multicolor; a black
       tee with orange sleeves has only one, so it is black). Colours of
       ACCENT_MIN % or more count too, whatever the model calls them. A
       striped item with a main colour is never multicolor: its stripes are
       a pattern, not extra colours.
    2a. A striped item with no main colour is multicolor with MULTI_COUNT or
       more listed colours, however thin the stripes.
    2. Otherwise MULTI_COUNT or more standout colours make it multicolor.
       Shades count separately: white + light blue + dark blue is three,
       and so are light, mid and dark grey (grey camo is multicolor).
    3. Otherwise the biggest colour names the item.
    """
    entries = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("color", "")).lower().strip()
        share = entry.get("share")
        if name in MULTI_COLORS and isinstance(share, (int, float)) and share > 0:
            entries.append((name, share, entry.get("kind")))
    if not entries:
        return None, None  # needs review
    # Rescale so shares add up to 100 (the model's rough shares often don't)
    total = sum(share for _, share, _ in entries)
    entries = [(name, share * 100 / total, kind == "panel" or share * 100 / total >= PANEL_SHARE)
               for name, share, kind in entries]
    entries.sort(key=lambda e: e[1], reverse=True)
    top_name, top_share, _ = entries[0]
    standouts = [name for name, share, _ in entries if share >= STANDOUT_MIN]
    distinct = list(dict.fromkeys(standouts))  # shades of one colour count once
    other_panels = {name for name, share, panel in entries[1:]
                    if panel and share >= PANEL_MIN and name != top_name}
    other_accents = {name for name, share, _ in entries[1:]
                     if share >= ACCENT_MIN and name != top_name}
    if top_share >= MAIN_MIN:
        if pattern != "striped" and len(other_panels | other_accents) >= PANEL_COUNT:
            return "multicolor", distinct[:4]
    elif pattern == "striped" and len(entries) >= MULTI_COUNT:
        # no main colour: every stripe colour counts, however thin
        return "multicolor", list(dict.fromkeys(name for name, _, _ in entries))[:4]
    elif len(standouts) >= MULTI_COUNT:
        return "multicolor", distinct[:4]
    return top_name, None


# ---------------------------------------------------------------------------
# 5. Save tags + vector.
#    Table:  CREATE EXTENSION vector;
#            CREATE TABLE items (id uuid PRIMARY KEY, image_path text,
#              original_filename text, category text, subcategory text, color text,
#              colors text[], pattern text, formality text, clip_vec vector(512));
#    Existing table:  ALTER TABLE items ADD COLUMN colors text[];
# ---------------------------------------------------------------------------
def save_item(item_id, image_path, original_filename, tags, vector):
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        conn.execute(
            psycopg.sql.SQL(
                """INSERT INTO {} (id, image_path, original_filename, category,
                                  subcategory, color, colors, pattern, formality, clip_vec)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            ).format(psycopg.sql.Identifier(ITEMS_TABLE)),
            (item_id, image_path, original_filename, tags["category"], tags["subcategory"],
             tags["color"], tags["colors"], tags["pattern"], tags["formality"], vector),
        )


# ---------------------------------------------------------------------------
# 6. The worker itself: what runs after each upload.
#    Progress is only printed to the terminal (nothing extra is stored in the database):
#      TAGGING -> TAGGED -> UPLOADED, or TAGGING_FAILED / UPLOAD_FAILED
# ---------------------------------------------------------------------------
def tagging_worker(item_id: str, image_path: str,
                   original_filename: str, media_type: str):
    # Background tasks fail silently from the client's view, so log the traceback.
    print(f"[TAGGING] {item_id} ({original_filename}) with {OPENAI_MODEL}")
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vector = clip_image_vector(img)
    except Exception:
        traceback.print_exc()
        print(f"[TAGGING_FAILED] {item_id} ({original_filename})")
        return

    try:
        tags = validate(tag_with_openai(image_bytes, media_type))
        print(f"[TAGGED] {item_id}: {tags}")
    except Exception:
        # OpenAI down or broken: keep the vector, leave tags for review
        traceback.print_exc()
        tags = validate({})
        print(f"[TAGGING_FAILED] {item_id}: openai failed, saving with empty tags")

    try:
        save_item(item_id, image_path, original_filename, tags, vector)
    except Exception:
        traceback.print_exc()
        print(f"[UPLOAD_FAILED] {item_id} ({original_filename})")
        return
    print(f"[UPLOADED] {item_id} saved to database")


# ---------------------------------------------------------------------------
# 7. Upload endpoint: saves the photo, replies instantly, tags in the background.
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app):
    load_clip()  # server loads CLIP once at start, not on the first upload
    yield


app = FastAPI(lifespan=lifespan)
os.makedirs("uploads", exist_ok=True)


@app.post("/items")
async def upload_item(file: UploadFile, background: BackgroundTasks):
    item_id = str(uuid.uuid4())
    image_path = f"uploads/{item_id}_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    background.add_task(tagging_worker, item_id, image_path,
                        file.filename, file.content_type or "image/jpeg")
    return {"item_id": item_id, "status": "tagging"}


# ---------------------------------------------------------------------------
# 8. Tag an image that is already on disk, without copying it to uploads/:
#      python tagging_worker_4.py birefnet_generate/birefnet_1_white_plain_tee.png
#    The path you pass is stored in the database as-is.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import mimetypes
    import sys

    for image_path in sys.argv[1:]:
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        tagging_worker(str(uuid.uuid4()), image_path,
                       os.path.basename(image_path), media_type)
