"""
Tagging worker for a wardrobe app (version 2: fully local, no APIs).

Runs once per uploaded clothing photo:
  1. Tags it (category, subcategory, color, pattern, formality)
  2. Computes its CLIP image vector (used later by the outfit engine for search)
  3. Saves both to the database

Everything runs on your own machine. No API keys, no per-image cost.

Tagging modes (set TAGGER below):
  "ollama" - small open-source vision model running locally via Ollama (default)
  "clip"   - CLIP zero-shot, fastest but weakest on formality

Install:
  pip install fastapi uvicorn python-multipart torch transformers pillow \
              psycopg[binary] pgvector python-dotenv ollama

Ollama setup (for TAGGER = "ollama"):
  1. Install Ollama from https://ollama.com
  2. ollama pull qwen2.5vl:3b          (or gemma3:4b, moondream, etc.)
  3. Make sure Ollama is running (the desktop app, or `ollama serve`)

Env vars:
  DATABASE_URL        - Postgres with the pgvector extension
  TAGGER              - optional, "ollama" (default) or "clip"
  OLLAMA_MODEL        - optional, local vision model name (default below)

Run:
  uvicorn tagging_worker_2:app --reload
"""

import io
import json
import os
import traceback
import uuid

import psycopg
import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, UploadFile
from pgvector.psycopg import register_vector
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()  # uvicorn doesn't read .env on its own

# Which tagger to use: "ollama" or "clip"
TAGGER = os.environ.get("TAGGER", "ollama")

# Which table to save to (same schema as items). Lets a one-off batch run
# write to a separate table without touching the live upload endpoint.
ITEMS_TABLE = os.environ.get("ITEMS_TABLE", "items")

# ---------------------------------------------------------------------------
# 1. The allowed tags. Every tagging option can only pick from these lists.
# ---------------------------------------------------------------------------
ALLOWED = {
    "category":    ["top", "bottom", "dress", "outerwear", "footwear"],
    "subcategory": ["t-shirts", "polos", "shirts", "hoodies, sweatshirts & jackets",
                     "shorts", "jeans", "sweatpants",
                     "blazers & suits", "coats", "cardigans & jumpers",
                     "shoes"],
    "color":       ["black", "white", "grey", "navy", "blue", "beige", "brown",
                     "green", "red", "orange", "pink", "yellow", "purple", "multicolor"],
    "pattern":     ["solid", "striped", "checked", "floral", "graphic", "other"],
    "formality":   ["casual", "smart casual", "business", "formal"],
}

# Instructions for the local vision model (Ollama).
PROMPT = (
    "Classify this clothing item. Reply with ONLY a JSON object with keys "
    "category, subcategory, color, pattern, formality. Each value must be one of:\n"
    + json.dumps(ALLOWED, indent=2)
)

# JSON schema built from ALLOWED, so the local model can only answer with
# allowed values.
TAG_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string", "enum": v} for k, v in ALLOWED.items()},
    "required": list(ALLOWED),
}

# CLIP matches sentences to images, so some tags need a clearer sentence than
# the default "a photo of a {option} {attribute} clothing item".
CLIP_PHRASES = {}

# ---------------------------------------------------------------------------
# 2. Load CLIP once when the server starts (not on every upload).
#    It's needed in every mode, because the outfit engine uses its vectors.
#    Tip: "patrickjohncyh/fashion-clip" is a fashion-tuned drop-in (also 512
#    dims). If you switch, re-embed items already saved with the old model.
# ---------------------------------------------------------------------------
CLIP_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_NAME).eval()
clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)


def _as_tensor(out):
    """Newer transformers versions return an output object instead of a tensor."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def clip_image_vector(img: Image.Image) -> list[float]:
    """The item's location on CLIP's 'meaning map' (512 numbers)."""
    with torch.no_grad():
        inputs = clip_proc(images=img, return_tensors="pt")
        vec = _as_tensor(clip_model.get_image_features(**inputs))
        vec = vec / vec.norm(dim=-1, keepdim=True)  # normalize for cosine search
    return vec[0].tolist()


# ---------------------------------------------------------------------------
# 3a. Tagging option A: local vision model via Ollama (free, open-source).
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")


def tag_with_ollama(image_bytes: bytes) -> dict:
    import ollama  # imported here so the other modes don't need it

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [image_bytes]}],
        format=TAG_SCHEMA,              # output must match the allowed values
        options={"temperature": 0},     # same photo -> same tags
    )
    return json.loads(resp["message"]["content"])


# ---------------------------------------------------------------------------
# 3b. Tagging option B: CLIP zero-shot (free, fastest, runs on your server).
# ---------------------------------------------------------------------------
def tag_with_clip(img: Image.Image) -> dict:
    tags = {}
    with torch.no_grad():
        img_inputs = clip_proc(images=img, return_tensors="pt")
        img_vec = _as_tensor(clip_model.get_image_features(**img_inputs))
        img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)

        for attribute, options in ALLOWED.items():
            # Turn each option into a sentence, e.g. "a photo of a striped garment"
            custom = CLIP_PHRASES.get(attribute, {})
            phrases = [custom.get(opt, f"a photo of a {opt} {attribute} clothing item")
                       for opt in options]
            txt_inputs = clip_proc(text=phrases, return_tensors="pt", padding=True)
            txt_vecs = _as_tensor(clip_model.get_text_features(**txt_inputs))
            txt_vecs = txt_vecs / txt_vecs.norm(dim=-1, keepdim=True)

            scores = (img_vec @ txt_vecs.T)[0]          # similarity to each phrase
            tags[attribute] = options[int(scores.argmax())]  # closest phrase wins
    return tags


def tag_image(img: Image.Image, image_bytes: bytes) -> dict:
    """Send the photo to whichever local tagger TAGGER selects."""
    if TAGGER == "ollama":
        return tag_with_ollama(image_bytes)
    if TAGGER == "clip":
        return tag_with_clip(img)
    raise ValueError(f"Unknown TAGGER {TAGGER!r}: use 'ollama' or 'clip'")


# ---------------------------------------------------------------------------
# 4. Validate: never trust model output blindly.
# ---------------------------------------------------------------------------
def validate(tags: dict) -> dict:
    clean = {}
    for attribute, options in ALLOWED.items():
        value = str(tags.get(attribute, "")).lower().strip()
        clean[attribute] = value if value in options else None  # None = needs review
    return clean


# ---------------------------------------------------------------------------
# 5. Save tags + vector.
#    Table:  CREATE EXTENSION vector;
#            CREATE TABLE items (id uuid PRIMARY KEY, image_path text,
#              original_filename text, category text, subcategory text, color text,
#              pattern text, formality text, clip_vec vector(512));
# ---------------------------------------------------------------------------
def save_item(item_id, image_path, original_filename, tags, vector):
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        conn.execute(
            psycopg.sql.SQL(
                """INSERT INTO {} (id, image_path, original_filename, category,
                                  subcategory, color, pattern, formality, clip_vec)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            ).format(psycopg.sql.Identifier(ITEMS_TABLE)),
            (item_id, image_path, original_filename, tags["category"], tags["subcategory"],
             tags["color"], tags["pattern"], tags["formality"], vector),
        )


# ---------------------------------------------------------------------------
# 6. The worker itself: what runs after each upload.
#    Progress is only printed to the terminal (nothing extra is stored in the database):
#      TAGGING -> TAGGED -> UPLOADED, or TAGGING_FAILED / UPLOAD_FAILED
# ---------------------------------------------------------------------------
def tagging_worker(item_id: str, image_path: str,
                   original_filename: str, media_type: str):
    # Background tasks fail silently from the client's view, so log the traceback.
    print(f"[TAGGING] {item_id} ({original_filename}) with {TAGGER}")
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tags = validate(tag_image(img, image_bytes))
        if tags["category"] == "footwear":  # only subcategory that applies to footwear
            tags["subcategory"] = "shoes"
        vector = clip_image_vector(img)  # always computed: the outfit engine needs it
    except Exception:
        traceback.print_exc()
        print(f"[TAGGING_FAILED] {item_id} ({original_filename})")
        return
    print(f"[TAGGED] {item_id}: {tags}")

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
app = FastAPI()
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
#      python tagging_worker_2.py gpt_generate/gpt_black_baggy_jeans_20260927_043218.png
#    The path you pass is stored in the database as-is.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import mimetypes
    import sys

    for image_path in sys.argv[1:]:
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        tagging_worker(str(uuid.uuid4()), image_path,
                       os.path.basename(image_path), media_type)
