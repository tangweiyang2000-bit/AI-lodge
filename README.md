# Shuffle

**An AI outfit recommender for your own wardrobe.**

A personal wardrobe app. Photograph each clothing item once: the app cuts out the
background, tags the item automatically (category, colour, pattern, formality…), then
suggests 3 outfits from a plain-language request such as
*"dinner date tonight, walking around outside"*. It learns from your 👍 / 👎 votes and
from what you actually wear.

## How it works

1. **Add an item** – upload a photo. The background is removed (macOS Vision, plus
   Real-ESRGAN upscaling for small photos), GPT-4.1 mini tags it, and FashionCLIP
   stores an image vector so similar-looking items can be found. You check the tags
   before it joins the wardrobe.
2. **Ask for outfits** – GPT turns your request into filters, Postgres + CLIP pick a
   shortlist of matching items, GPT builds 3 outfits from that shortlist, and code
   checks every outfit against the rules and your past feedback.
3. **Give feedback** – 👍 / 👎 (with a reason) and "I wore this" are saved and used in
   the next request.

## Project layout

| Path | What's in it |
|---|---|
| `outfit_engine.py` | The app: outfit API, feedback, history, and serves the website. Also a CLI. |
| `tagging_worker.py` | Upload + auto-tagging API, the allowed tag values and tag rules. |
| `retag_items.py` | Re-tags every item with the current tagger. |
| `smoke_test.py` | End-to-end check against a running server. |
| `web/` | The website (Today, Wardrobe, Item, Add an item, History). |
| `sql/` | Database schema (`schema.sql`) and one-off migrations. |
| `macos_generate/` | Background removal on macOS, plus the cut-outs made from the sample photos. |
| `test_images/` | Sample photos: the 60 test items used to build and try the app. |

## Running it

Needs macOS, Python 3.14, PostgreSQL with the `pgvector` extension, Xcode Command Line
Tools (to build the background-removal helper), and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

createdb wardrobe
psql -d wardrobe -f sql/schema.sql

# .env (never commit it)
#   OPENAI_API_KEY=...
#   DATABASE_URL=postgresql://localhost/wardrobe

uvicorn outfit_engine:app --reload --host 0.0.0.0
# open http://127.0.0.1:8000/
```

The first upload compiles the Swift helper and downloads Real-ESRGAN (~150 MB) into
`macos_generate/bin/`.

Check everything works (with the server running):

```bash
python smoke_test.py
```
