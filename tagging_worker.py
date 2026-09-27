"""
Tagging worker for a wardrobe app.

Runs once per uploaded clothing photo:
  1. Tags it (category, color, pattern, formality, warmth)
  2. Computes its CLIP image vector (used later by the outfit engine for search)
  3. Saves both to the database

Install:
  pip install fastapi uvicorn python-multipart openai torch transformers pillow psycopg[binary] pgvector

Env vars:
  OPENAI_API_KEY      - for the LLM tagging option
  OPENAI_MODEL        - optional, any vision-capable GPT model (default below)
  DATABASE_URL        - Postgres with the pgvector extension

Run:
  uvicorn tagging_worker:app --reload
"""

import base64
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

# ---------------------------------------------------------------------------
# 1. The allowed tags. Both tagging options can only pick from these lists.
# ---------------------------------------------------------------------------
ALLOWED = {
    "category":  ["top", "bottom", "dress", "outerwear", "shoes"],
    "color":     ["black", "white", "grey", "navy", "blue", "beige", "brown",
                  "green", "red", "pink", "yellow", "purple", "multicolor"],
    "pattern":   ["solid", "striped", "checked", "floral", "graphic", "other"],
    "formality": ["casual", "smart casual", "business", "formal"],
    # How warm the piece is. Replaces season, which doesn't fit Singapore:
    #   light  = fine outdoors in the heat (linen, thin cotton, shorts, sandals)
    #   medium = a layer for air-conditioned places (cardigan, overshirt, jeans)
    #   heavy  = only for travel somewhere cold (wool coat, puffer, thick knit)
    "warmth":    ["light", "medium", "heavy"],
}

# CLIP matches sentences to images, so some tags need a clearer sentence than
# the default "a photo of a {option} {attribute} clothing item".
CLIP_PHRASES = {
    "warmth": {
        "light":  "a photo of lightweight, thin, breathable clothing for hot weather",
        "medium": "a photo of medium-weight clothing, like a cardigan or light jacket",
        "heavy":  "a photo of heavy, thick, warm clothing for cold winter weather",
    },
}

# ---------------------------------------------------------------------------
# 2. Load CLIP once when the server starts (not on every upload).
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
# 3a. Tagging option A: CLIP zero-shot (free, runs on your server).
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


# ---------------------------------------------------------------------------
# 3b. Tagging option B: multimodal LLM (better on formality, costs per call).
# ---------------------------------------------------------------------------
llm = OpenAI()  # reads OPENAI_API_KEY
# Any GPT model that accepts images works; check OpenAI's model list for current names.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def tag_with_llm(image_bytes: bytes, media_type: str) -> dict:
    prompt = (
        "Classify this clothing item. Reply with ONLY a JSON object with keys "
        "category, color, pattern, formality, warmth. Each value must be one of:\n"
        + json.dumps(ALLOWED, indent=2)
        + "\n\nWarmth means how warm the item is to wear: light = fine outdoors in "
        "tropical heat, medium = a layer for air-conditioned rooms, "
        "heavy = only for cold climates."
    )
    # OpenAI takes images as a data URL: "data:image/jpeg;base64,...."
    data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"

    resp = llm.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=200,
        response_format={"type": "json_object"},  # forces valid JSON back
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
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
    return clean


# ---------------------------------------------------------------------------
# 5. Save tags + vector.
#    Table:  CREATE EXTENSION vector;
#            CREATE TABLE items (id uuid PRIMARY KEY, image_path text,
#              original_filename text, category text, color text, pattern text, formality text,
#              warmth text, clip_vec vector(512));
# ---------------------------------------------------------------------------
def save_item(item_id, image_path, original_filename, tags, vector):
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        conn.execute(
            """INSERT INTO items (id, image_path, original_filename, category,
                                  color, pattern, formality, warmth, clip_vec)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (item_id, image_path, original_filename, tags["category"], tags["color"],
             tags["pattern"], tags["formality"], tags["warmth"], vector),
        )


# ---------------------------------------------------------------------------
# 6. The worker itself: what runs after each upload.
#    Progress is only printed to the terminal (nothing extra is stored in the database):
#      TAGGING -> TAGGED -> UPLOADED, or TAGGING_FAILED / UPLOAD_FAILED
# ---------------------------------------------------------------------------
USE_LLM = True  # flip to False to use free CLIP tagging instead

def tagging_worker(item_id: str, image_path: str,
                   original_filename: str, media_type: str):
    # Background tasks fail silently from the client's view, so log the traceback.
    print(f"[TAGGING] {item_id} ({original_filename})")
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        raw_tags = tag_with_llm(image_bytes, media_type) if USE_LLM else tag_with_clip(img)
        tags = validate(raw_tags)
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
#      python tagging_worker.py gpt_generate/gpt_black_baggy_jeans_20260927_043218.png
#    The path you pass is stored in the database as-is.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import mimetypes
    import sys

    for image_path in sys.argv[1:]:
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        tagging_worker(str(uuid.uuid4()), image_path,
                       os.path.basename(image_path), media_type)
