-- Feedback loop for the Today page. Safe to re-run.

-- Your 👍 / 👎 on outfits. An outfit is its exact set of clothes (outfit_key = the item
-- ids, sorted and joined). A vote belongs to an occasion (its dress level, e.g. "casual"),
-- because an outfit that's wrong for the gym can be right for brunch.
CREATE TABLE IF NOT EXISTS outfit_feedback (
  outfit_key text NOT NULL,                -- sorted item ids joined with ','
  occasion   text NOT NULL,                -- dress level(s) read from the request, e.g. 'casual|smart casual'
  item_ids   uuid[] NOT NULL,
  vote       smallint NOT NULL CHECK (vote IN (-1, 1)),   -- 1 = 👍, -1 = 👎
  reason     text,                         -- 👎 reason chip, e.g. 'colours clash' (optional)
  request    text,                         -- what you asked for
  title      text,                         -- the outfit's title at the time
  voted_at   timestamptz NOT NULL DEFAULT now(),
  last_shown timestamptz NOT NULL DEFAULT now(),   -- liked outfits repeat at most once a week
  PRIMARY KEY (outfit_key, occasion)
);

-- "I wore this": one row per outfit per day you wore it.
CREATE TABLE IF NOT EXISTS outfit_wears (
  outfit_key text NOT NULL,
  worn_on    date NOT NULL DEFAULT current_date,
  item_ids   uuid[] NOT NULL,
  request    text,
  title      text,
  PRIMARY KEY (outfit_key, worn_on)
);
