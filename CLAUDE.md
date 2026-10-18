# Wardrobe AI Outfit Recommender

A personal wardrobe app: photograph every clothing item once, the system removes the
background and tags it automatically, then suggests 3 outfits from a plain-language
request ("dinner date tonight, walking around outside") and learns from your feedback.
Single user, Singapore. Full requirements: `Wardrobe AI Outfit Recommender — PRD.docx`
(21 Sep 2026).

## About the developer

I'm new to machine learning and still learning Python. When working with me:
- Explain ML terms in plain language the first time they come up.
- Prefer simple, readable code with comments over clever or compact code.
- Make one change at a time and tell me how to check it worked.
- Ask before deciding anything that changes how the app behaves for me.
- No hardcoded per-item rules; keep guides and rules generic.

## Status

| PRD phase | Status |
|---|---|
| 1. Manual bootstrap | Done (superseded by auto-tagging) |
| 2. Outfit engine | Done: `outfit_engine.py` (always aims for 3 outfits, swap one piece) |
| 3. Auto-tagging | Done: `tagging_worker.py`, 60 items tagged |
| 4. Real UI | Done: website in `web/` (Today, Wardrobe, Item, Add an item, History) |
| 5. Feedback & polish | Done: 👍/👎 with reasons, "I wore this", laundry, history, not-worn-lately |

**Possible next steps (not started):** dresses/skirts in the tag lists, putting the app
online (works only on this Mac / home Wi-Fi today), weather awareness.
**Not planned:** calendar integration, season tags (Singapore has no seasons).

## How it works

```
Photo(s) ──► Add an item page ──► POST /items ──► tagging_worker()  (4 at a time)
                                                   │ make_upright (phone EXIF rotation)
                                                   │ remove_background (macOS Vision + Real-ESRGAN)
                                                   │ GPT-4.1 mini tags + validate() rules
                                                   │ FashionCLIP vector (clip_vec)
                                                   ▼
                                  uploads/pending/<id>.json   (waits for your check)
                                                   │ "Add to wardrobe" = POST /items/{id}/confirm
                                                   ▼
                                           Postgres `items` table (read via view_items)
                                                   │
Request ──► Today page ──► POST /outfits ──► 1. Parse: GPT turns the request into filters
                                             2. Retrieve: SQL filters, CLIP ranks per slot
                                             3. Generate: GPT builds outfits from the shortlist,
                                                told your past likes / dislikes / recent wears
                                             4. Validate: code rejects invented ids, rule breaks,
                                                blocked dislikes; tops up to 3 (ask again, then
                                                fill from the rules)
                                                   ▼
                                     3 outfits ──► 👍 / 👎 (+reason) / I wore this
                                                   ▼
                                     outfit_feedback + outfit_wears tables ──► back into step 3
```

## Decisions already made

- **Tagging model: GPT-4.1 mini at `temperature=0`, `seed=42`.** Chosen over GPT-5 mini
  because it answers the same way every time: 59/60 items identical over 5 runs vs
  53/60, and 1.1 s vs 2.6 s per call. GPT-5 models don't accept `temperature`; the code
  switches to `reasoning_effort="low"` for them.
- **The model describes, code applies the rules.** GPT can only answer with allowed
  values (strict JSON schema). `validate()` enforces the fixed rules. GPT lists colour
  shares; code decides the main colour.
- **Tried and reverted, don't reintroduce without asking:** measuring colour from
  pixels, majority voting over several GPT calls, a Hugging Face colour classifier
  (40/60), a draggable flat-lay, a background-"dissolve" animation, "Looks similar" on
  the item page, uploading photos before you press Upload.
- **Each item is tagged once and stored.** Re-tag only deliberately with `retag_items.py`.
- **Uploads are held until you confirm.** Website uploads go to `uploads/pending/` and
  only enter `items` (and the outfits) when you press Add to wardrobe. The CLI
  (`python tagging_worker.py <photo>`) still saves straight away.
- **The name you type is stored as `original_filename`** (with the photo's extension),
  because that's what the website and outfit engine show as the item's name.
- **Outfit engine: GPT-4o mini** for parsing and generation, with Postgres + CLIP
  retrieval in between so GPT only sees a shortlist (`PER_CATEGORY = 6` per slot).
- **Feedback rules** (`judge()` in `outfit_engine.py`):
  - 👎 "colours clash" / "don't like a piece" → that outfit never comes back.
  - Any other 👎 (too formal, too casual, not for this occasion, no reason) → blocked
    only for the same occasion (its formality level), e.g. fine for brunch, not the gym.
  - 👍 and "I wore this" → the AI leans towards them. A liked outfit may repeat, at most
    one per request, and not within 7 days of last being shown or worn.
  - Pieces worn in the last 3 days are avoided where possible.
- **No outerwear category.** Items are top / bottom / footwear; hoodies, sweatshirts and
  jackets can also fill the outerwear slot (`DUAL_SLOT_SUBCATEGORIES`).
- **Background removal: macOS Vision** (`macos_generate/`), plus Real-ESRGAN 4x
  upscaling for small photos, onto a transparent 1024×1536 canvas. macOS-only, so
  `birefnet_generate/use_birefnet_generate.py` (rembg + BiRefNet, runs on Linux) is kept
  as the option if the app moves to a cloud server.
- **Thumbnails:** grids use `/thumb/<path>` (400px WebP, made on first request, cached in
  `thumbs/`); the outfit board and item page use the full photo.
- **Pretrained models only, no custom training** (PRD non-goal for v1).

## Conventions

### Tag values (`ALLOWED` in `tagging_worker.py`, the single source)
| Tag | Values |
|---|---|
| category | top, bottom, footwear |
| subcategory | t-shirts, polos, shirts, hoodies, sweatshirts, jackets, shorts, jeans, sweatpants, shoes, sandals |
| color | black, white, grey, blue, beige, brown, green, red, orange, pink, yellow, purple, multicolor |
| pattern | solid, striped, checked, dotted, floral, graphic, camo (+ none for footwear) |
| formality | casual, smart casual, business casual, formal |
| sleeve | short, long, none |
| shade | light, dark |

`outfit_engine.py` imports `ALLOWED`. Never copy the lists elsewhere.

### Rules set by code in `validate()`
- Shoes or sandals → category footwear; footwear → pattern `none`.
- Bottoms and footwear → sleeve `none`.
- All polos → formality smart casual.
- One colour ≥ 60 % of the item names it; otherwise the multicolor rules in
  `color_from_shares()`. Camo is always multicolor.
- Any value outside `ALLOWED` becomes `None` (needs review).

### Meanings that matter
- **shade:** light = white, cream, pastel or pale shades incl. light grey; dark =
  everything else, incl. bright colours and all browns.
- **colors:** comma-separated text, filled only when `color` is multicolor, else `none`.
- **occasion** (feedback): the request's formality levels joined, e.g. `casual|smart casual`.

### Images
Cut-outs of the original 60 live in `macos_generate/` as transparent 1024×1536 PNGs
named `macos_<n>_<colour>_<description>.png`. Website uploads live in `uploads/`
(`<id>_<file>` original, `<id>_cutout.png` cut-out).

## Database

PostgreSQL `wardrobe` on localhost with pgvector. Full schema in `db_creation`.

**`items`** (one row per clothing item; read through the view `view_items`)
| Column | Type | Notes |
|---|---|---|
| id | uuid PK | |
| image_path | text | relative to the project folder |
| original_filename | text | the item's display name |
| category, subcategory, color, colors, pattern, formality, sleeve, shade | text | tags above |
| clip_vec | vector(512) | FashionCLIP (`patrickjohncyh/fashion-clip`) image embedding |
| available | boolean | false = in the laundry; outfits skip it |

**`outfit_feedback`** (`add_feedback.sql`): PK (outfit_key, occasion); item_ids uuid[],
vote (1 / -1), reason, request, title, voted_at, last_shown.
**`outfit_wears`** (`add_feedback.sql`): PK (outfit_key, worn_on date); item_ids, request, title.
`outfit_key` = the outfit's item ids sorted and joined with commas.

## Rules that must not be broken

- **`view_items` must stay a VIEW**, never a table. Re-run
  `CREATE OR REPLACE VIEW view_items AS SELECT * FROM items;` after adding a column.
- **A schema change touches:** `db_creation`, the INSERT in `save_item()`,
  `ITEM_FIELDS` in `tagging_worker.py`, and `retag_items.py`'s `TAG_COLUMNS`.
- **Image and search vectors must use the same CLIP model.** Changing `CLIP_NAME` means
  re-embedding every item.
- **Back up before re-tagging:** `CREATE TABLE items_backup AS TABLE items;` then
  `retag_items.py`. Try tagging changes on a few photos with the CLI first.
- **Never trust GPT output blindly:** tags go through `validate()`, outfits through
  `validate_outfits()` (which also applies the feedback rules).
- **Never commit `.env`** (API keys). It's in `.gitignore`.
- **After changing a page, bump `?v=` on `style.css` / `common.js`** in every HTML file,
  so browsers fetch the new version.

## Files

| File | Purpose |
|---|---|
| `tagging_worker.py` | Tagger + FastAPI app. Item endpoints: `POST /items` (upload, held for review), `GET /items`, `GET/PATCH/DELETE /items/{id}`, `POST /items/{id}/confirm`, `GET /allowed`. Defines `ALLOWED`, the prompt, schema and `validate()`. |
| `outfit_engine.py` | Outfit engine on the same app: `POST /outfits`, `POST /outfits/swap`, `POST /feedback`, `POST /wears`, `GET /history`, `GET /feedback`, `GET /thumb/<path>`; serves the website at `/`. Also a CLI. |
| `web/` | The website: `index.html` (Today), `wardrobe.html`, `item.html`, `add.html`, `history.html`, `style.css`, `common.js`, `gsap.min.js` (animations). |
| `macos_generate/use_macos_generate.py` | Background removal (`make_cutout`), also run in batch on its own. |
| `birefnet_generate/use_birefnet_generate.py` | Alternative cut-out pipeline (rembg + BiRefNet) that also runs on Linux; not used by the app today. Downloads its model on first run. |
| `retag_items.py` | Re-tags every row with the current tagger (tag columns only). |
| `smoke_test.py` | Checks a running server end to end (read-only by default). |
| `db_creation` | Full schema: `items`, `view_items`, `outfit_feedback`, `outfit_wears`. |
| `add_available.sql`, `add_feedback.sql` | One-off migrations (already applied / to apply once). |
| `view_items.sql` | Inspection query over the view. |
| `outfits.html` | Preview of the last outfits, rewritten on every request (git-ignored). |
| `test_images/` | Raw source photos. |

Git: one branch only, `main`, pushed to GitHub (`tangweiyang2000-bit/AI-lodge`).
Commit straight to `main`; no other branches or worktrees.

## Commands

```bash
source .venv/bin/activate

# the app (website + API); --host lets your phone on the same Wi-Fi open it
uvicorn outfit_engine:app --reload --host 0.0.0.0
#   Mac:   http://127.0.0.1:8000/
#   phone: http://<ipconfig getifaddr en0>:8000

# check everything works (server must be running)
python smoke_test.py

# outfits from the terminal (opens outfits.html)
python outfit_engine.py "dinner date tonight, walking around outside"

# tag one image already on disk (saved straight to the wardrobe)
python tagging_worker.py macos_generate/macos_1_white_plain_tee.png

# re-tag the whole table (back it up first)
python retag_items.py

# database migrations (psql needs the database name; $DATABASE_URL is only in .env)
psql -d wardrobe -f add_feedback.sql
```

### Environment (`.env`)
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `DATABASE_URL`, `GEMINI_API_KEY`, `GEMINI_BASE_URL`.
Optional: `OPENAI_MODEL`, `ITEMS_TABLE`, `FEEDBACK_TABLE`, `WEARS_TABLE`, `UPLOAD_WORKERS`
(photos processed at once, default 4).

**Gotcha:** both the tagger and the outfit engine read `OPENAI_MODEL`. Setting it changes
both models (tagger default `gpt-4.1-mini`, engine default `gpt-4o-mini`).

## Glossary

- **Tagging:** labelling each item with category, colour, pattern and so on.
- **Embedding / vector (`clip_vec`):** 512 numbers describing what an image looks like;
  similar-looking items have similar vectors.
- **CLIP / FashionCLIP:** a model that puts images and text on the same "meaning map", so
  a style sentence can be compared with item photos.
- **Cosine distance (`<=>`):** how far apart two vectors point; smaller = more similar.
- **Temperature:** how random a model's answers are; 0 = as repeatable as possible.
- **JSON schema (strict):** a fixed answer format the model must follow, with allowed values.
- **Feedback loop:** your votes and wears are stored and fed back into the next request.
- **Thumbnail:** a small copy of a photo for grids, so pages load fast.
