CREATE EXTENSION IF NOT EXISTS vector;

-- One row per clothing item, written by tagging_worker.py (13 columns).
CREATE TABLE IF NOT EXISTS items (
  id uuid PRIMARY KEY,
  image_path text,
  original_filename text,
  category text,          -- top | bottom | footwear
  subcategory text,       -- t-shirts, polos, shirts, hoodies, sweatshirts, jackets,
                          -- shorts, jeans, sweatpants, shoes, sandals
  color text,             -- main colour, or 'multicolor'
  colors text,            -- comma-separated colours when multicolor, else 'none'
  pattern text,           -- solid | striped | checked | dotted | floral | graphic | camo
  formality text,         -- casual | smart casual | business casual | formal
  sleeve text,            -- short | long | none
  shade text,             -- light | dark
  clip_vec vector(512),   -- CLIP image embedding
  available boolean NOT NULL DEFAULT true  -- false = in the laundry / unavailable; outfits skip it
);

-- Databases created from the older 9-column version: add the newer columns.
-- Safe to re-run; does nothing if the columns already exist.
ALTER TABLE items ADD COLUMN IF NOT EXISTS colors text;
ALTER TABLE items ADD COLUMN IF NOT EXISTS sleeve text;
ALTER TABLE items ADD COLUMN IF NOT EXISTS shade text;
ALTER TABLE items ADD COLUMN IF NOT EXISTS available boolean NOT NULL DEFAULT true;

-- Re-run the CREATE OR REPLACE VIEW below after adding a column: a SELECT * view
-- keeps the column list it had when it was created.
-- view_items is the read view outfit_engine.py queries by default (ITEMS_TABLE).
-- It must stay a VIEW, not a table: tagging_worker.py only ever writes to `items`,
-- so a separate table here would silently go stale.
CREATE OR REPLACE VIEW view_items AS SELECT * FROM items;

-- Feedback loop (sql/add_feedback.sql): your thumbs up/down on outfits (per occasion, with an
-- optional reason) and the outfits you wore. The outfit engine reads both.
CREATE TABLE IF NOT EXISTS outfit_feedback (
  outfit_key text NOT NULL,                -- sorted item ids joined with ','
  occasion   text NOT NULL,                -- dress level(s), e.g. 'casual|smart casual'
  item_ids   uuid[] NOT NULL,
  vote       smallint NOT NULL CHECK (vote IN (-1, 1)),   -- 1 = up, -1 = down
  reason     text,                         -- e.g. 'colours clash', 'too formal'
  request    text,
  title      text,
  voted_at   timestamptz NOT NULL DEFAULT now(),
  last_shown timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (outfit_key, occasion)
);

CREATE TABLE IF NOT EXISTS outfit_wears (
  outfit_key text NOT NULL,
  worn_on    date NOT NULL DEFAULT current_date,
  item_ids   uuid[] NOT NULL,
  request    text,
  title      text,
  PRIMARY KEY (outfit_key, worn_on)
);
