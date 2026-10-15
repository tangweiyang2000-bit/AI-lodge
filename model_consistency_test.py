"""
Model consistency test: GPT-4.1 mini at temperature 0 vs GPT-5 mini (your current tagger).

Both models get exactly the same prompt, JSON schema, image size and validate() rules
from tagging_worker_4.py. Only the model (and its randomness setting) differs:
  - gpt-4.1-mini: temperature=0, seed=42  (randomness turned off)
  - gpt-5-mini:   reasoning_effort="low"   (as in tagging_worker_4.py; no temperature option)

Reads images only: nothing is written to the database, the tagger is not changed.

Run (from the project folder, venv active):
    python model_consistency_test.py                 # both models, 5 runs each
    python model_consistency_test.py 5 gpt-4.1-mini  # one model only

Writes model_runs.csv (every answer) and prints, per model:
  consistent = items with identical tags on every run
  agreement  = runs whose colour matches your current tags in view_items_2.csv
"""
import base64
import csv
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tagging_worker_4 as tw

HERE = Path(__file__).resolve().parent
FIELDS = ["category", "subcategory", "color", "colors", "pattern", "formality", "sleeve", "shade"]
WORKERS = 8

MODEL_SETTINGS = {
    "gpt-4.1-mini": {"temperature": 0, "seed": 42},
    "gpt-4.1": {"temperature": 0, "seed": 42},
    "gpt-5-mini": {"reasoning_effort": "low"},
}


def tag(model: str, image_bytes: bytes) -> dict:
    data_url = "data:image/png;base64," + base64.b64encode(tw.shrink(image_bytes)).decode()
    resp = tw.llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": tw.PROMPT},
        ]}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "tags", "schema": tw.TAG_SCHEMA, "strict": True}},
        **MODEL_SETTINGS.get(model, {}),
    )
    return tw.validate(json.loads(resp.choices[0].message.content))


def run_one(job):
    model, name, run = job
    t0 = time.time()
    try:
        t = tag(model, (HERE / "macos_generate" / name).read_bytes())
        t["colors"] = ", ".join(sorted(t["colors"])) if t["colors"] else "none"  # compare as a set
        t["error"] = ""
    except Exception as e:
        traceback.print_exc()
        t = {f: "" for f in FIELDS} | {"error": repr(e)}
    t["seconds"] = round(time.time() - t0, 2)
    return model, name, run, t


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    models = sys.argv[2:] or ["gpt-4.1-mini", "gpt-5-mini"]
    current = {r["original_filename"]: r for r in csv.DictReader(open(HERE / "view_items_2.csv"))}
    names = list(current)
    jobs = [(m, n, r) for m in models for n in names for r in range(1, runs + 1)]
    print(f"{len(names)} items x {runs} runs x {len(models)} models = {len(jobs)} calls ...")

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(run_one, jobs))

    with open(HERE / "model_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "file", "run", *FIELDS, "current_color", "seconds", "error"])
        for m, n, r, t in results:
            w.writerow([m, n, r, *[t.get(k, "") for k in FIELDS], current[n]["color"],
                        t["seconds"], t["error"]])

    for m in models:
        by_item = defaultdict(list)
        for mm, n, r, t in results:
            if mm == m:
                by_item[n].append(t)
        print(f"\n=== {m} ===")
        consistent = colour_consistent = colour_agree = calls = errors = 0
        secs = []
        for n in names:
            ts = by_item[n]
            errors += sum(1 for t in ts if t["error"])
            secs += [t["seconds"] for t in ts if not t["error"]]
            wobbly = {f: Counter(t.get(f, "") for t in ts) for f in FIELDS}
            wobbly = {f: c for f, c in wobbly.items() if len(c) > 1}
            consistent += not wobbly
            colour_consistent += "color" not in wobbly and "colors" not in wobbly
            colour_agree += sum(t.get("color") == current[n]["color"] for t in ts)
            calls += len(ts)
            if wobbly:
                parts = "; ".join(f"{f}: " + ", ".join(f"{v} x{k}" for v, k in c.most_common())
                                  for f, c in wobbly.items())
                print(f"  {n}: {parts}")
        print(f"  all tags consistent:   {consistent}/{len(names)} items")
        print(f"  colours consistent:    {colour_consistent}/{len(names)} items")
        print(f"  colour = current tag:  {colour_agree}/{calls} answers")
        print(f"  errors: {errors} | avg {sum(secs) / max(len(secs), 1):.1f}s per call")
    print("\nSaved model_runs.csv")


if __name__ == "__main__":
    main()
