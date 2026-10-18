"""
Tagging worker for a wardrobe app (version 4: GPT tags, CLIP vectors).

Runs once per uploaded clothing photo:
  1. Tags it with OpenAI GPT-5 mini (category, subcategory, color, pattern, formality)
  2. Computes its FashionCLIP image vector (used later by the outfit engine for search)
  3. Saves both to the database

What's new in version 4:
  - Tags come from OpenAI (cloud API) instead of CLIP zero-shot / Ollama.
    The model's answer is locked to the allowed values with a JSON schema.
  - CLIP is only used for the search vector
  - If OpenAI fails, the item is still saved with its vector and empty tags
    (None = needs review), instead of being lost.

Env vars:
  DATABASE_URL        - Postgres with the pgvector extension
  OPENAI_API_KEY      - OpenAI API key
  OPENAI_MODEL        - optional, default "gpt-4.1-mini"
  ITEMS_TABLE         - optional, table to write to (default "items")

"""

import base64
import contextlib
import functools
import io
import json
import os
import threading
import traceback
import uuid

import psycopg
import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from openai import OpenAI
from pgvector.psycopg import register_vector
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

load_dotenv()  # uvicorn doesn't read .env on its own

# Which table to save to (same schema as items). Lets a one-off batch run
# write to a separate table without touching the live upload endpoint.
ITEMS_TABLE = os.environ.get("ITEMS_TABLE", "items")

# ---------------------------------------------------------------------------
# 1. The allowed tags. The model can only pick from these lists.
# ---------------------------------------------------------------------------
ALLOWED = {
    "category":    ["top", "bottom", "footwear"],
    "subcategory": ["t-shirts", "polos", "shirts", "hoodies", "sweatshirts", "jackets",
                    "shorts", "jeans", "sweatpants",
                    "shoes", "sandals"],
    "color":       ["black", "white", "grey", "blue", "beige", "brown",
                    "green", "red", "orange", "pink", "yellow", "purple",
                    "multicolor"],
    "pattern":     ["solid", "striped", "checked", "dotted", "floral", "graphic", "camo"],
    "formality":   ["casual", "smart casual", "business casual", "formal"],
    "sleeve":      ["short", "long", "none"],
    "shade":       ["light", "dark"],
}

# Only the borderline shades that tend to get mislabelled.
COLOR_GUIDE = {
    "black":  "incl. washed black and very dark, faded denim",
    "grey":   "incl. charcoal, slate, heather",
    "white":  "incl. off-white, cream, ivory, chalk, and warm-toned or aged off-white",
    "blue":   "incl. navy, indigo, mid-blue, light blue",
    "beige":  "only pale sand or cream-tan",
    "brown":  "incl. tan, camel, taupe, khaki, chocolate",
    "green":  "incl. olive, teal, blue-green, forest, bottle",
    "red":    "incl. raspberry and bright or deep pinkish-red",
    "pink":   "only pale or pastel pink",
    "purple": "incl. plum",
}


FOOTWEAR_SUBCATEGORIES = ("shoes", "sandals")

# Only the subcategories that tend to get mixed up.
SUBCATEGORY_GUIDE = {
    "shoes":       "closed-toe, incl. sneakers, loafers and clogs",
    "sandals":     "open footwear, incl. slides and flip-flops",
    "polos":       "collar with a short button placket, short or long sleeve, incl. rugby shirts",
    "shirts":      "buttons all the way down the front",
    "jackets":     "zip-up or button-front, no hood; a tall collar may look like a hood",
    "hoodies":     "only with a real hood, often with drawstrings",
    "sweatshirts": "pullover, no hood, no zip",
}

# Light vs dark: only pale shades count as light (they show stains).
SHADE_GUIDE = {
    "light": "white, cream, pastel or pale shades, incl. light grey",
    "dark":  "everything else, incl. bright saturated colours and all browns, even taupe",
}

# Only the patterns that tend to get mixed up.
PATTERN_GUIDE = {
    "graphic": "large printed or embroidered logo, text or image",
    "dotted":  "incl. pin dots, tiny, faint or tone-on-tone dot prints, even on dark fabric",
    "solid":   "no print, incl. colour-blocked panels and small logos",
}

# Only the formality levels that tend to get mixed up.
FORMALITY_GUIDE = {
    "smart casual": "polished but relaxed, incl. all loafers and polos",
    "business casual": "incl. button-down shirts",
}

# Instructions for the model.
PROMPT = (
    "Classify this clothing item. Each value must be one of:\n"
    + json.dumps(ALLOWED)
    + "\nSubcategory notes: " + "; ".join(f"{k} {v}" for k, v in SUBCATEGORY_GUIDE.items())
    + ".\nShade notes: " + "; ".join(f"{k} {v}" for k, v in SHADE_GUIDE.items())
    + ".\nPattern notes: " + "; ".join(f"{k} {v}" for k, v in PATTERN_GUIDE.items())
    + ".\nFormality notes: " + "; ".join(f"{k} {v}" for k, v in FORMALITY_GUIDE.items())
    + ".\nColour notes: " + "; ".join(f"{k} {v}" for k, v in COLOR_GUIDE.items())
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
        **{k: {"type": "string", "enum": v} for k, v in ALLOWED.items() if k not in ("color", "shade")},
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
        # shade comes after colors so the model judges it with the colours already named
        # (before, it called mid-brown loafers light)
        "shade": {"type": "string", "enum": ALLOWED["shade"]},
    },
    "required": [*[k for k in ALLOWED if k not in ("color", "shade")], "colors", "shade"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# 2. Load CLIP once when the server starts (not on every upload).
#    outfit_engine.py embeds its search text with this same model
#    (load_clip) and compares it against the vectors stored here.
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
# 3. Tagging: OpenAI (GPT-4.1 mini at temperature 0 by default).
# ---------------------------------------------------------------------------
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
# GPT-5 models don't take temperature; others run with randomness off
# (model_consistency_test.py: gpt-4.1-mini 59/60 consistent vs 53/60 for gpt-5-mini).
if OPENAI_MODEL.startswith("gpt-5"):
    MODEL_SETTINGS = {"reasoning_effort": "low"}
else:
    MODEL_SETTINGS = {"temperature": 0, "seed": 42}
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
        **MODEL_SETTINGS,
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
    # Footwear subcategories and the footwear category always go together
    # (tags can disagree), defaulting to shoes
    if clean["subcategory"] in FOOTWEAR_SUBCATEGORIES:
        clean["category"] = "footwear"
    elif clean["category"] == "footwear":
        clean["subcategory"] = "shoes"
    if clean["category"] in ("bottom", "footwear"):
        clean["sleeve"] = "none"  # only tops have sleeves
    clean["color"], clean["colors"] = color_from_shares(tags.get("colors"), clean["pattern"])
    # Camo is always multicolor, even when the model lists its shades as one colour
    if clean["pattern"] == "camo" and clean["color"] is not None:
        names = [str(c.get("color")).lower() for c in tags.get("colors") or [] if isinstance(c, dict)]
        clean["color"] = "multicolor"
        clean["colors"] = list(dict.fromkeys(n for n in names if n in MULTI_COLORS))[:4] or None
    if clean["category"] == "footwear":
        clean["pattern"] = "none"  # footwear has no pattern, like bottoms have no sleeve
    if clean["subcategory"] == "polos":
        clean["formality"] = "smart casual"  # all polos, incl. rugby shirts
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
#              colors text, pattern text, formality text, sleeve text, shade text,
#              clip_vec vector(512));
#    Existing table:  ALTER TABLE items ADD COLUMN colors text;
#                     ALTER TABLE items ADD COLUMN sleeve text;
#                     ALTER TABLE items ADD COLUMN shade text;
# ---------------------------------------------------------------------------
def save_item(item_id, image_path, original_filename, tags, vector):
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        conn.execute(
            psycopg.sql.SQL(
                """INSERT INTO {} (id, image_path, original_filename, category,
                                  subcategory, color, colors, pattern, formality, sleeve, shade, clip_vec)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            ).format(psycopg.sql.Identifier(ITEMS_TABLE)),
            (item_id, image_path, original_filename, tags["category"], tags["subcategory"],
             tags["color"], ", ".join(tags["colors"]) if tags["colors"] else "none",
             tags["pattern"], tags["formality"], tags["sleeve"], tags["shade"], vector),
        )


# ---------------------------------------------------------------------------
# 6. The worker itself: what runs after each upload.
#    Progress is only printed to the terminal (nothing extra is stored in the database):
#      TAGGING -> TAGGED -> UPLOADED, or TAGGING_FAILED / UPLOAD_FAILED
# ---------------------------------------------------------------------------
def remove_background(item_id: str, image_path: str) -> str:
    """Cut the garment out of an uploaded photo with macos_generate's pipeline
    (macOS Vision + Real-ESRGAN) and save it as uploads/<id>_cutout.png.
    Returns the cut-out's path, or the original path if anything goes wrong,
    so a failed cut-out never loses the item."""
    try:
        from macos_generate.use_macos_generate import make_cutout
        cutout_path = f"uploads/{item_id}_cutout.png"
        make_cutout(os.path.abspath(image_path)).save(cutout_path)
        print(f"[BACKGROUND_REMOVED] {item_id} -> {cutout_path}")
        return cutout_path
    except Exception:
        traceback.print_exc()
        print(f"[BACKGROUND_KEPT] {item_id}: background removal failed, using the original photo")
        return image_path


# Website uploads wait here until you press "Add to wardrobe" (POST /items/{id}/confirm):
# uploads/pending/<id>.json holds the tags and vector, so nothing reaches the
# items table (or the outfits) before you've checked it. Survives a server restart.
PENDING_DIR = os.path.join("uploads", "pending")


def pending_path(item_id) -> str:
    return os.path.join(PENDING_DIR, f"{item_id}.json")


def hold_for_review(item_id, image_path, original_filename, tags, vector):
    os.makedirs(PENDING_DIR, exist_ok=True)
    with open(pending_path(item_id), "w") as f:
        json.dump({"id": str(item_id), "image_path": image_path, "original_filename": original_filename,
                   "tags": tags, "vector": [float(v) for v in vector], "available": True}, f)


def load_pending(item_id) -> dict | None:
    try:
        with open(pending_path(item_id)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def pending_item(item_id) -> dict | None:
    """A held upload in the same shape as GET /items/{id}, plus "pending": true."""
    held = load_pending(item_id)
    if held is None:
        return None
    item = {"id": held["id"], "image_path": held["image_path"], "original_filename": held["original_filename"],
            **{f: held["tags"].get(f) for f in TAG_FIELDS}, "colors": held["tags"].get("colors") or [],
            "available": held.get("available", True), "pending": True}
    return item


# The website uploads a whole batch at once. Up to UPLOAD_WORKERS photos (default 4) are
# processed side by side: most of the time is spent waiting on OpenAI and the cut-out tool,
# which are fine in parallel. The two steps that share one resource take turns instead:
# CLIP (one model in memory, _clip_lock below) and upscaling (the GPU, in use_macos_generate).
_workers = threading.BoundedSemaphore(int(os.environ.get("UPLOAD_WORKERS", "4")))
_clip_lock = threading.Lock()


def tagging_worker(item_id: str, image_path: str,
                   original_filename: str, media_type: str, cut_out: bool = False, hold: bool = False):
    with _workers:
        _tagging_worker(item_id, image_path, original_filename, media_type, cut_out, hold)


def _tagging_worker(item_id: str, image_path: str,
                    original_filename: str, media_type: str, cut_out: bool = False, hold: bool = False):
    # Background tasks fail silently from the client's view, so log the traceback.
    print(f"[TAGGING] {item_id} ({original_filename}) with {OPENAI_MODEL}")
    if cut_out:  # uploads: tag and store the cut-out, like the items in macos_generate/
        image_path = remove_background(item_id, image_path)
        if image_path.endswith("_cutout.png"):
            media_type = "image/png"
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with _clip_lock:
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

    if hold:   # website upload: wait for "Add to wardrobe"
        try:
            hold_for_review(item_id, image_path, original_filename, tags, vector)
        except Exception:
            traceback.print_exc()
            print(f"[UPLOAD_FAILED] {item_id} ({original_filename})")
            return
        print(f"[READY_FOR_REVIEW] {item_id}: waiting for Add to wardrobe")
        return
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


def display_filename(name: str | None, filename: str) -> str:
    """The name you typed becomes the item's original_filename (keeping the photo's
    extension), because that's what the website and outfit engine show as its name.
    No name typed: the photo's own file name, as before."""
    name = " ".join((name or "").replace("/", " ").replace("\\", " ").split())[:60]
    if not name:
        return filename
    return name + os.path.splitext(filename)[1].lower()


def make_upright(image_path: str) -> None:
    """Phone cameras save photos sideways plus an EXIF note saying "rotate me".
    Browsers obey the note, but the cut-out tool (macOS Vision) and CLIP read the raw
    pixels, so the garment came out sideways. Turn the pixels the right way up once,
    here, and drop the note. Photos without the note are left untouched."""
    try:
        with Image.open(image_path) as img:
            if img.getexif().get(0x0112, 1) == 1:   # 0x0112 = Orientation; 1 = already upright
                return
            fmt = img.format
            upright = ImageOps.exif_transpose(img)
        if fmt == "JPEG":
            upright.convert("RGB").save(image_path, "JPEG", quality=95)
        else:
            upright.save(image_path, fmt)
        print(f"[ROTATED] {image_path} turned upright")
    except Exception:
        traceback.print_exc()   # e.g. a format PIL can't read: carry on with the photo as it is


@app.post("/items")
async def upload_item(file: UploadFile, background: BackgroundTasks, name: str | None = Form(None)):
    item_id = str(uuid.uuid4())
    image_path = f"uploads/{item_id}_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    make_upright(image_path)

    background.add_task(tagging_worker, item_id, image_path, display_filename(name, file.filename),
                        file.content_type or "image/jpeg", True, True)  # remove background, hold for review
    return {"item_id": item_id, "status": "tagging"}


# ---------------------------------------------------------------------------
# 7b. Read and edit items (for the UI): list them, get one, change tags or
#     availability by hand. Manual edits only check the values are allowed;
#     the validate() rules are for the model's answers, not yours.
# ---------------------------------------------------------------------------
ITEM_FIELDS = ["id", "image_path", "original_filename", "category", "subcategory", "color",
               "colors", "pattern", "formality", "sleeve", "shade", "available"]
TAG_FIELDS = ["category", "subcategory", "color", "pattern", "formality", "sleeve", "shade"]


def item_from_row(row) -> dict:
    item = dict(zip(ITEM_FIELDS, row))
    item["id"] = str(item["id"])
    # colors is stored as "blue, yellow" or "none"; the API uses a list
    item["colors"] = [] if item["colors"] in (None, "", "none") else item["colors"].split(", ")
    return item


def select_items(where: str = "", params: tuple = ()) -> list[dict]:
    query = psycopg.sql.SQL("SELECT {} FROM {} " + where + " ORDER BY original_filename").format(
        psycopg.sql.SQL(", ").join(map(psycopg.sql.Identifier, ITEM_FIELDS)),
        psycopg.sql.Identifier(ITEMS_TABLE))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        return [item_from_row(r) for r in conn.execute(query, params).fetchall()]


WEARS_TABLE = os.environ.get("WEARS_TABLE", "outfit_wears")   # "I wore this" (add_feedback.sql)


def with_last_worn(items: list[dict]) -> list[dict]:
    """Add "last_worn" (YYYY-MM-DD, or None if never) to each item, from the wear history."""
    worn = {}
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            worn = {str(i): d.isoformat() for i, d in conn.execute(psycopg.sql.SQL(
                "SELECT item_id, max(worn_on) FROM {}, unnest(item_ids) AS item_id GROUP BY item_id")
                .format(psycopg.sql.Identifier(WEARS_TABLE))).fetchall()}
    except psycopg.errors.UndefinedTable:
        pass   # add_feedback.sql not run yet: nothing has been worn
    for item in items:
        item["last_worn"] = worn.get(item["id"])
    return items


@app.get("/items")
def list_items():
    """Every item with its tags, availability and when it was last worn, for the Wardrobe screen."""
    return with_last_worn(select_items())


@app.get("/items/{item_id}")
def get_item(item_id: uuid.UUID):
    """One item. Uploads not added yet come back with "pending": true; 404 = unknown or still tagging."""
    items = select_items("WHERE id = %s", (item_id,))
    if items:
        return with_last_worn(items)[0]
    held = pending_item(item_id)
    if held is None:
        raise HTTPException(404, "No item with that id")
    return held


class ConfirmBody(BaseModel):
    name: str | None = None   # the name typed on the Add page (blank = keep the photo's file name)


@app.post("/items/{item_id}/confirm")
def confirm_item(item_id: uuid.UUID, body: ConfirmBody | None = None):
    """"Add to wardrobe": move a checked upload from uploads/pending/ into the items table,
    with the name you typed while it was being processed."""
    held = load_pending(item_id)
    if held is None:
        if select_items("WHERE id = %s", (item_id,)):
            return select_items("WHERE id = %s", (item_id,))[0]   # already added
        raise HTTPException(404, "No upload waiting with that id")
    name = display_filename(body.name if body else None, held["original_filename"])
    save_item(item_id, held["image_path"], name, held["tags"], held["vector"])
    if not held.get("available", True):
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            conn.execute(psycopg.sql.SQL("UPDATE {} SET available = false WHERE id = %s").format(
                psycopg.sql.Identifier(ITEMS_TABLE)), (item_id,))
    os.remove(pending_path(item_id))
    print(f"[UPLOADED] {item_id} added to the wardrobe")
    return select_items("WHERE id = %s", (item_id,))[0]


class ItemUpdate(BaseModel):
    """Only the fields you send are changed; leave the rest out."""
    category: str | None = None
    subcategory: str | None = None
    color: str | None = None
    colors: list[str] | None = None
    pattern: str | None = None
    formality: str | None = None
    sleeve: str | None = None
    shade: str | None = None
    available: bool | None = None


@app.patch("/items/{item_id}")
def update_item(item_id: uuid.UUID, body: ItemUpdate):
    """Change some of an item's tags and/or its availability, e.g. {"color": "green"}
    or {"available": false} when it goes in the laundry."""
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(400, "Send at least one field to change")

    raw_colors = None   # the colors list as sent, for uploads not added yet
    # Every value must be one of the allowed tags (pattern may also be "none", as for footwear)
    for field in TAG_FIELDS:
        if field in changes:
            allowed = ALLOWED[field] + (["none"] if field == "pattern" else [])
            if changes[field] not in allowed:
                raise HTTPException(422, f"{field} must be one of {allowed}")
    if "colors" in changes:
        bad = [c for c in changes["colors"] if c not in MULTI_COLORS]
        if bad:
            raise HTTPException(422, f"colors must come from {MULTI_COLORS}, not {bad}")
        raw_colors = changes["colors"]
        changes["colors"] = ", ".join(changes["colors"]) or "none"
    # colors only means something for multicolor items
    if "color" in changes and changes["color"] != "multicolor":
        changes["colors"] = "none"

    query = psycopg.sql.SQL("UPDATE {} SET {} WHERE id = %s RETURNING {}").format(
        psycopg.sql.Identifier(ITEMS_TABLE),
        psycopg.sql.SQL(", ").join(psycopg.sql.SQL("{} = %s").format(psycopg.sql.Identifier(f))
                                   for f in changes),
        psycopg.sql.SQL(", ").join(map(psycopg.sql.Identifier, ITEM_FIELDS)))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute(query, (*changes.values(), item_id)).fetchone()
    if row is not None:
        print(f"[EDITED] {item_id}: {changes}")
        return item_from_row(row)

    # Not in the wardrobe yet: edit the held upload instead
    held = load_pending(item_id)
    if held is None:
        raise HTTPException(404, "No item with that id")
    for field in TAG_FIELDS:
        if field in changes:
            held["tags"][field] = changes[field]
    if "colors" in changes:
        held["tags"]["colors"] = raw_colors if changes["colors"] != "none" else None
    if "available" in changes:
        held["available"] = changes["available"]
    with open(pending_path(item_id), "w") as f:
        json.dump(held, f)
    print(f"[EDITED] {item_id} (not added yet): {changes}")
    return pending_item(item_id)


@app.delete("/items/{item_id}")
def delete_item(item_id: uuid.UUID):
    """Remove an item from the wardrobe for good. Its uploaded photos (uploads/<id>_*)
    are deleted too; photos elsewhere, like macos_generate/, are never touched."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute(psycopg.sql.SQL("DELETE FROM {} WHERE id = %s RETURNING image_path").format(
            psycopg.sql.Identifier(ITEMS_TABLE)), (item_id,)).fetchone()
    if row is None and load_pending(item_id) is None:
        raise HTTPException(404, "No item with that id")
    if load_pending(item_id) is not None:   # an upload you decided not to add
        os.remove(pending_path(item_id))
    removed = []
    if os.path.isdir("uploads"):
        for name in os.listdir("uploads"):
            if name.startswith(f"{item_id}_"):
                os.remove(os.path.join("uploads", name))
                removed.append(name)
    print(f"[DELETED] {item_id} (photos removed: {removed or 'none'})")
    return {"deleted": str(item_id)}


@app.get("/allowed")
def allowed_values():
    """The allowed tag values, so the UI's dropdowns use the same lists as the tagger."""
    return {**ALLOWED, "multi_colors": MULTI_COLORS}


@app.get("/items/{item_id}/similar")
def similar_items(item_id: uuid.UUID, n: int = 3):
    """The n items whose CLIP vectors are closest to this one ("Looks similar")."""
    table = psycopg.sql.Identifier(ITEMS_TABLE)
    fields = psycopg.sql.SQL(", ").join(map(psycopg.sql.Identifier, ITEM_FIELDS))
    query = psycopg.sql.SQL(
        """SELECT {fields}, 1 - (clip_vec <=> (SELECT clip_vec FROM {t} WHERE id = %s)) AS score
           FROM {t}
           WHERE id <> %s AND clip_vec IS NOT NULL
           ORDER BY clip_vec <=> (SELECT clip_vec FROM {t} WHERE id = %s)
           LIMIT %s""").format(fields=fields, t=table)
    if not select_items("WHERE id = %s", (item_id,)):
        raise HTTPException(404, "No item with that id")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(query, (item_id, item_id, item_id, max(1, min(n, 12)))).fetchall()
    return [item_from_row(r[:-1]) | {"score": round(float(r[-1]), 2)} for r in rows]


# ---------------------------------------------------------------------------
# 8. Tag an image that is already on disk, without copying it to uploads/:
#      python tagging_worker.py birefnet_generate/birefnet_1_white_plain_tee.png
#    The path you pass is stored in the database as-is.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import mimetypes
    import sys

    for image_path in sys.argv[1:]:
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        tagging_worker(str(uuid.uuid4()), image_path,
                       os.path.basename(image_path), media_type)
