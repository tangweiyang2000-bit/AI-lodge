"""Export view_items_2 to view_items_2.csv (same columns/order as view_items_2.sql),
plus a readable summary of each item's CLIP vector:
  has_vector  - whether the item was embedded
  vec_dims    - vector length (should be 512)
  vec_preview - first 5 values of the vector
  similar_1..3 - the 3 most similar other items by cosine similarity (score in brackets)
"""
import csv
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

QUERY = """
select v.original_filename, v.category, v.subcategory, v.sleeve, v.color, v.shade,
       v.colors, v.pattern, v.formality, v.image_path,
       v.clip_vec is not null                      as has_vector,
       vector_dims(v.clip_vec)                     as vec_dims,
       v.clip_vec::text                            as vec_text,
       (select array_agg(s.original_filename || ' (' || round(s.sim::numeric, 3) || ')'
                         order by s.sim desc)
          from (select o.original_filename, 1 - (o.clip_vec <=> v.clip_vec) as sim
                  from view_items_2 o
                 where o.id <> v.id and o.clip_vec is not null
                 order by o.clip_vec <=> v.clip_vec
                 limit 3) s)                       as similar
from view_items_2 v
order by split_part(v.original_filename, '_', 2)::int
"""

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    cur = conn.execute(QUERY)
    rows = cur.fetchall()

header = ["original_filename", "category", "subcategory", "sleeve", "color", "shade",
          "colors", "pattern", "formality", "image_path",
          "has_vector", "vec_dims", "vec_preview", "similar_1", "similar_2", "similar_3"]

out = HERE / "view_items_2.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for *base, has_vec, dims, vec_text, similar in rows:
        if vec_text:
            first5 = [float(x) for x in vec_text.strip("[]").split(",")[:5]]
            preview = "[" + ", ".join(f"{x:.3f}" for x in first5) + ", …]"
        else:
            preview = ""
        similar = (similar or []) + [""] * 3
        w.writerow([*base, has_vec, dims, preview, *similar[:3]])

print(f"Wrote {len(rows)} rows to {out}")
