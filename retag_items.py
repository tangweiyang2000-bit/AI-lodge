"""Re-tag every row already in ITEMS_TABLE with the current tagger in tagging_worker_4.py
(same model, prompt, guides and validate() rules as a new upload).

Only the tag columns are updated; id, image_path, original_filename and clip_vec stay as they are.
Back up the table first, e.g.  CREATE TABLE items_backup AS TABLE items;

Run (from the project folder, venv active):
    python retag_items.py
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
from psycopg import sql

import tagging_worker_4 as tw

HERE = Path(__file__).resolve().parent
WORKERS = 8
TAG_COLUMNS = ["category", "subcategory", "color", "colors", "pattern", "formality", "sleeve", "shade"]


def retag(row):
    item_id, image_path, name = row
    tags = tw.validate(tw.tag_with_openai((HERE / image_path).read_bytes(), "image/png"))
    tags["colors"] = ", ".join(tags["colors"]) if tags["colors"] else "none"
    return item_id, name, tags


def main():
    table = sql.Identifier(tw.ITEMS_TABLE)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(sql.SQL("SELECT id, image_path, original_filename FROM {}").format(table)).fetchall()
        old = {r[0]: r[1:] for r in conn.execute(
            sql.SQL("SELECT id, {} FROM {}").format(
                sql.SQL(", ").join(map(sql.Identifier, TAG_COLUMNS)), table)).fetchall()}
        print(f"Re-tagging {len(rows)} rows in {tw.ITEMS_TABLE} with {tw.OPENAI_MODEL} ...")

        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(retag, rows))

        update = sql.SQL("UPDATE {} SET {} WHERE id = %s").format(
            table, sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(c)) for c in TAG_COLUMNS))
        changed = 0
        for item_id, name, tags in sorted(results, key=lambda r: int(r[1].split("_")[1])):
            new = tuple(tags[c] for c in TAG_COLUMNS)
            conn.execute(update, (*new, item_id))
            diffs = [f"{c}: {o} -> {n}" for c, o, n in zip(TAG_COLUMNS, old[item_id], new) if o != n]
            if diffs:
                changed += 1
                print(f"  {name:45} " + "; ".join(diffs))
    print(f"Done: {changed}/{len(rows)} rows changed")


if __name__ == "__main__":
    main()
