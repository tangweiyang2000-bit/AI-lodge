-- Step 1, part 1: add the `available` column (false = in the laundry / unavailable).

-- 1. Back up the table first (an exact copy you can restore from)
CREATE TABLE IF NOT EXISTS items_backup_before_available AS TABLE items;

-- 2. Add the column: every existing item starts as available
ALTER TABLE items ADD COLUMN IF NOT EXISTS available boolean NOT NULL DEFAULT true;

-- 3. Rebuild the view so it includes the new column
CREATE OR REPLACE VIEW view_items AS SELECT * FROM items;

-- Check: should show one row, true | 60
SELECT available, COUNT(*) FROM view_items GROUP BY available;
