"""
Tagging worker for a wardrobe app (version 3: fully local, no APIs, hybrid tagging).

Runs once per uploaded clothing photo:
  1. Tags it (category, subcategory, color, pattern, formality)
  2. Computes its CLIP image vector (used later by the outfit engine for search)
  3. Saves both to the database

Everything runs on your own machine. No API keys, no per-image cost.

What's new in version 3:
  - "hybrid" mode (default): CLIP tags first, and only the tags CLIP is unsure
    about are re-asked to the local vision model (Ollama/Qwen).
  - CLIP now reports a confidence (0-1) for every tag.
  - Natural CLIP sentences for every option (CLIP_PHRASES), instead of
    awkward ones like "a photo of a casual formality clothing item".
  - CLIP sentence vectors are computed once at startup, not on every upload.
  - The image vector is computed once and reused for tagging AND search.
  - If Ollama is down in hybrid mode, CLIP's guesses are kept instead of failing.

Tagging modes (set TAGGER below):
  "hybrid" - CLIP first, Ollama only for unsure tags (default)
  "ollama" - small open-source vision model for every tag
  "clip"   - CLIP zero-shot only, fastest but weakest on formality

Install:
  pip install fastapi uvicorn python-multipart torch transformers pillow \
              psycopg[binary] pgvector python-dotenv ollama

Ollama setup (for TAGGER = "hybrid" or "ollama"):
  1. Install Ollama from https://ollama.com
  2. ollama pull qwen2.5vl:3b          (or gemma3:4b, moondream, etc.)
  3. Make sure Ollama is running (the desktop app, or `ollama serve`)

Env vars:
  DATABASE_URL        - Postgres with the pgvector extension
  TAGGER              - optional, "hybrid" (default), "ollama" or "clip"
  OLLAMA_MODEL        - optional, local vision model name (default below)
  ITEMS_TABLE         - optional, table to write to (default "items")

Run:
  uvicorn tagging_worker_3:app --reload
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

# Which tagger to use: "hybrid", "ollama" or "clip"
TAGGER = os.environ.get("TAGGER", "hybrid")

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
                     "blazers & suits", "coats",
                     "shoes"],
    "color":       ["black", "white", "grey", "blue", "beige", "brown",
                     "green", "red", "orange", "pink", "yellow", "purple",
                     "multicolor"],
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

# ---------------------------------------------------------------------------
# 2. CLIP sentences. CLIP matches whole sentences to images, so each option
#    gets a natural description. Options not listed here fall back to
#    "a photo of a {option} piece of clothing" (fine for colors).
# ---------------------------------------------------------------------------
CLIP_PHRASES = {
    "category": {
        "top":       "a photo of a top, such as a t-shirt or shirt",
        "bottom":    "a photo of trousers, jeans or shorts",
        "dress":     "a photo of a dress",
        "outerwear": "a photo of a jacket or coat",
        "footwear":  "a photo of a pair of shoes",
    },
    "subcategory": {
        "t-shirts": "a photo of a thin short-sleeved cotton t-shirt, no hood, no ribbed cuffs",
        "polos":    "a photo of a polo shirt with a collar and a short button placket",
        "shirts":   "a photo of a button-up shirt",
        "hoodies, sweatshirts & jackets": (
            "a photo of a pullover hoodie, sweatshirt, or crewneck worn on the upper "
            "body, or a casual zip-up jacket (bomber, varsity, denim, leather) - not "
            "a tailored blazer/suit, and not a long coat"
        ),
        "shorts":     "a photo of shorts",
        "jeans":      "a photo of denim jeans",
        "sweatpants": "a photo of sweatpants or jogger trousers worn on the legs",
        "blazers & suits": (
            "a photo of a tailored structured blazer or suit jacket with lapels, "
            "worn in business or formal settings"
        ),
        "coats": "a photo of a long coat extending past the hips, such as an overcoat or trench coat",
        "shoes": "a photo of a pair of shoes",
    },
    "color": {
        "black": [
            "a photo of a black piece of clothing",
            "a photo of slightly faded black clothing, still a deep dark shade",
            "a photo of a black garment that has patches or large areas "
            "of a bright contrasting color, while the main body of the "
            "garment is black",
            "faded, chalky black, clothing, still a deep, dark shade",
            "a photo of plain black clothing, extremely low saturation, "
            "with no printed image or graphic on it",
        ],
        "blue": [
            "a photo of blue clothing, from light blue to dark navy blue",
            "a photo of pale sky-blue clothing",
        ],
        "white": [
            "a photo of a white piece of clothing",
            "a photo of white colored footwear",
        ],
        "brown": [
            "a photo of a brown piece of clothing",
            "a photo of brown colored footwear",
        ],
        "green": [
            "a photo of a green piece of clothing",
            "faded, chalky, dusty forest green, clothing, still a deep, "
            "dark shade",
        ],
        "purple": "a photo of a purple piece of clothing",
        "grey": [
            "a photo of a grey piece of clothing",
            "a photo of grey clothing with a small brightly colored "
            "accent, grey is the main overall colour of the garment",
            "deep dark ash grey clothing with small brightly colored "
            "accent patches, ash grey is clearly the main and dominant "
            "overall colour",
        ],
        "multicolor": [
            "a photo of a multicolor piece of clothing",
            "a photo of a multicolored piece of clothing",
            "a photo of clothing with a complex mix of three or four "
            "different colors",
            "a photo of multicolor footwear",
            "a photo of sneaker footwear with several different colored "
            "panels and sections",
        ],
    },
    "pattern": {
        "solid":   "a photo of a plain, solid-colored piece of clothing",
        "striped": "a photo of a striped piece of clothing",
        "checked": "a photo of a checked or plaid piece of clothing",
        "floral":  "a photo of a piece of clothing with a floral print",
        "graphic": "a photo of a piece of clothing with a large printed graphic",
        "other":   "a photo of a piece of clothing with an unusual pattern",
    },
    "formality": {
        "casual":       "a photo of casual everyday clothing",
        "smart casual": "a photo of smart casual clothing",
        "business":     "a photo of business office clothing",
        "formal":       "a photo of formal wear, such as a suit or evening outfit",
    },
}

# Hybrid mode: if CLIP's confidence for a tag is below this, Qwen decides it.
# Confidence = CLIP's probability for its top choice (0-1), tuned per
# attribute. Formality is set high because CLIP is weakest there.
CLIP_MIN_CONF = {
    "category":    0.60,
    "subcategory": 0.65,
    "color":       0.50,
    "pattern":     0.60,
    "formality":   0.80,
}

# ---------------------------------------------------------------------------
# 3. Load CLIP once when the server starts (not on every upload).
#    It's needed in every mode, because the outfit engine uses its vectors.
#
#    EXPERIMENTAL: patrickjohncyh/fashion-clip (fashion-tuned, also 512 dims)
#    in place of openai/clip-vit-base-patch32 - a true drop-in since it's the
#    same CLIP ViT-B/32 architecture, just fine-tuned on fashion product
#    images/descriptions. Still re-embeds everything (the vector space moved),
#    so keep ITEMS_TABLE pointed at a separate table while experimenting.
# ---------------------------------------------------------------------------
CLIP_NAME = "patrickjohncyh/fashion-clip"
clip_model = CLIPModel.from_pretrained(CLIP_NAME).eval()
clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)


def _as_tensor(out):
    """Newer transformers versions return an output object instead of a tensor."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def _encode_phrases() -> dict:
    """Turn every CLIP sentence into a vector once, at startup.

    An option's phrase(s) in CLIP_PHRASES can be a single string or a list of
    strings (prompt ensembling): e.g. "black" needs both a "pure black" and a
    "washed/faded black" phrasing, since merging them into one sentence dilutes
    both (a crisp black item scores worse against the merged phrase than
    against "pure black" alone, and vice versa for washed black). Each option
    keeps its phrases as separate vectors; scoring takes the best match per
    option instead of averaging them into one blurred vector.
    """
    feats = {}
    with torch.no_grad():
        for attribute, options in ALLOWED.items():
            custom = CLIP_PHRASES.get(attribute, {})
            option_vecs = []
            for opt in options:
                entry = custom.get(opt, f"a photo of a {opt} piece of clothing")
                phrases = entry if isinstance(entry, list) else [entry]
                inputs = clip_proc(text=phrases, return_tensors="pt", padding=True)
                vecs = _as_tensor(clip_model.get_text_features(**inputs))
                option_vecs.append(vecs / vecs.norm(dim=-1, keepdim=True))
            feats[attribute] = option_vecs  # one (n_phrases, dim) tensor per option
    return feats


TEXT_FEATS = _encode_phrases()


def clip_image_tensor(img: Image.Image) -> torch.Tensor:
    """The item's location on CLIP's 'meaning map' (1 x 512, normalized)."""
    with torch.no_grad():
        inputs = clip_proc(images=img, return_tensors="pt")
        vec = _as_tensor(clip_model.get_image_features(**inputs))
        return vec / vec.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 4a. Tagging option A: local vision model via Ollama (free, open-source).
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b")


def tag_with_ollama(image_bytes: bytes) -> dict:
    import ollama  # imported here so CLIP-only mode doesn't need it

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [image_bytes]}],
        format=TAG_SCHEMA,              # output must match the allowed values
        options={"temperature": 0},     # same photo -> same tags
    )
    return json.loads(resp["message"]["content"])


# ---------------------------------------------------------------------------
# 4b. Tagging option B: CLIP zero-shot, with a confidence per tag.
# ---------------------------------------------------------------------------
def tag_with_clip(img_vec: torch.Tensor) -> tuple[dict, dict]:
    tags, conf = {}, {}
    with torch.no_grad():
        scale = clip_model.logit_scale.exp()  # CLIP's built-in sharpening factor
        for attribute, options in ALLOWED.items():
            # Each option may have several phrasings (see _encode_phrases) -
            # score it by whichever phrasing matches best, not their average.
            logits = torch.stack([
                (scale * img_vec @ vecs.T)[0].max() for vecs in TEXT_FEATS[attribute]
            ])
            probs = logits.softmax(dim=-1)
            best = int(probs.argmax())
            tags[attribute] = options[best]
            conf[attribute] = round(float(probs[best]), 3)
    return tags, conf


# ---------------------------------------------------------------------------
# 4c. Pick the tagger. Returns (tags, a short note for the terminal log).
# ---------------------------------------------------------------------------
def tag_image(img_vec: torch.Tensor, image_bytes: bytes) -> tuple[dict, str]:
    if TAGGER == "ollama":
        return tag_with_ollama(image_bytes), "all tags from ollama"
    if TAGGER not in ("clip", "hybrid"):
        raise ValueError(f"Unknown TAGGER {TAGGER!r}: use 'hybrid', 'ollama' or 'clip'")

    tags, conf = tag_with_clip(img_vec)
    if TAGGER == "clip":
        return tags, f"clip conf {conf}"

    unsure = [a for a, c in conf.items() if c < CLIP_MIN_CONF.get(a, 0.5)]
    if not unsure:
        return tags, f"all tags from clip, conf {conf}"

    try:
        llm_tags = tag_with_ollama(image_bytes)
    except Exception:
        # Ollama down or broken: keep CLIP's guesses rather than failing the upload
        traceback.print_exc()
        return tags, f"ollama failed, kept clip for {unsure}, conf {conf}"

    for attribute in unsure:
        if attribute in llm_tags:
            tags[attribute] = llm_tags[attribute]
    return tags, f"ollama decided {unsure}, clip conf {conf}"


# ---------------------------------------------------------------------------
# 5. Validate: never trust model output blindly.
# ---------------------------------------------------------------------------
def validate(tags: dict) -> dict:
    clean = {}
    for attribute, options in ALLOWED.items():
        value = str(tags.get(attribute, "")).lower().strip()
        clean[attribute] = value if value in options else None  # None = needs review
    # Footwear and "shoes" always go together (tags can come from two models)
    if clean["category"] == "footwear" or clean["subcategory"] == "shoes":
        clean["category"], clean["subcategory"] = "footwear", "shoes"
    return clean


# ---------------------------------------------------------------------------
# 6. Save tags + vector.
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
# 7. The worker itself: what runs after each upload.
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

        img_vec = clip_image_tensor(img)  # computed once: used for tags AND search
        raw_tags, note = tag_image(img_vec, image_bytes)
        tags = validate(raw_tags)
        vector = img_vec[0].tolist()
    except Exception:
        traceback.print_exc()
        print(f"[TAGGING_FAILED] {item_id} ({original_filename})")
        return
    print(f"[TAGGED] {item_id}: {tags}  ({note})")

    try:
        save_item(item_id, image_path, original_filename, tags, vector)
    except Exception:
        traceback.print_exc()
        print(f"[UPLOAD_FAILED] {item_id} ({original_filename})")
        return
    print(f"[UPLOADED] {item_id} saved to database")


# ---------------------------------------------------------------------------
# 8. Upload endpoint: saves the photo, replies instantly, tags in the background.
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
# 9. Tag an image that is already on disk, without copying it to uploads/:
#      python tagging_worker_3.py gpt_generate/gpt_black_baggy_jeans_20260927_043218.png
#    The path you pass is stored in the database as-is.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import mimetypes
    import sys

    for image_path in sys.argv[1:]:
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        tagging_worker(str(uuid.uuid4()), image_path,
                       os.path.basename(image_path), media_type)