"""Smoke test: checks the running app end to end, from the outside, like the website does.

Start the server first (uvicorn outfit_engine:app --reload), then in another terminal:
    python smoke_test.py            # read-only checks (nothing is changed)
    python smoke_test.py --write    # also saves a 👍 and an "I wore this", then removes them again

It asks for outfits and a swap, so it makes a few OpenAI calls (a few cents at most).
Every check prints PASS or FAIL with a short reason; the last line is the total.
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
WRITE = "--write" in sys.argv
results = []


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=120) as res:
        raw = res.read()
        return res.status, (json.loads(raw) if res.headers.get_content_type() == "application/json" else raw)


def check(name, fn):
    try:
        detail = fn()
        results.append(True)
        print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
        return True
    except urllib.error.HTTPError as err:
        msg = err.read().decode(errors="replace")[:200]
        print(f"FAIL  {name}  (HTTP {err.code}: {msg})")
    except Exception as err:  # noqa: BLE001 - report anything that goes wrong
        print(f"FAIL  {name}  ({type(err).__name__}: {err})")
    results.append(False)
    return False


state = {}


def website():
    status, page = call("GET", "/app/")
    assert status == 200 and b"Where are you off to?" in page, "Today page didn't load"
    for page in ("wardrobe.html", "item.html", "add.html", "history.html", "style.css", "common.js", "gsap.min.js"):
        assert call("GET", f"/app/{page}")[0] == 200, f"{page} missing"
    return "all 5 pages + styles + scripts"


def items():
    _, data = call("GET", "/items")
    assert data, "no items in the wardrobe"
    state["items"] = data
    untagged = [i["original_filename"] for i in data if not (i["category"] and i["color"])]
    assert "last_worn" in data[0], "last_worn missing (old tagging_worker.py?)"
    away = sum(1 for i in data if not i["available"])
    return f"{len(data)} items, {away} in the laundry" + (f", {len(untagged)} need tags: {untagged[:3]}" if untagged else "")


def one_item():
    first = state["items"][0]
    _, item = call("GET", f"/items/{first['id']}")
    assert item["id"] == first["id"]
    return item["original_filename"]


def allowed():
    _, data = call("GET", "/allowed")
    assert "category" in data and "multi_colors" in data
    return f"{len(data['subcategory'])} types, {len(data['color'])} colours"


def thumbnails():
    first = state["items"][0]
    status, body = call("GET", "/thumb/" + urllib.request.quote(first["image_path"]))
    assert status == 200 and len(body) < 200_000, "thumbnail missing or too big"
    return f"{len(body) // 1024} KB for {first['image_path'].split('/')[-1]}"


def history():
    _, data = call("GET", "/history")
    return f"{len(data['votes'])} votes, {len(data['wears'])} wears saved"


def outfits():
    _, data = call("POST", "/outfits", {"request": "casual lunch with friends", "n": 3})
    got = len(data.get("outfits", []))
    assert got >= 1, data.get("message", "no outfits")
    state["outfit"] = data["outfits"][0]
    state["constraints"] = data["constraints"]
    note = "" if got == 3 else f" (only {got}: the wardrobe may not fit 3 for this request)"
    return f"{got} outfits: " + ", ".join(o["title"] for o in data["outfits"]) + note


def swap():
    items_now = {slot: it["id"] for slot, it in state["outfit"]["items"].items()}
    slot = "footwear" if "footwear" in items_now else "top"
    _, data = call("POST", "/outfits/swap", {"request": "casual lunch with friends", "items": items_now,
                                             "slot": slot, "constraints": state["constraints"], "n": 3})
    for opt in data["options"]:
        kept = {s: i["id"] for s, i in opt["items"].items() if s != slot}
        assert kept == {s: i for s, i in items_now.items() if s != slot}, "swap changed other pieces"
    return f"{len(data['options'])} other {slot} options"


def feedback_round_trip():
    outfit = state["outfit"]
    ids = [i["id"] for i in outfit["items"].values()]
    common = {"item_ids": ids, "formality": state["constraints"]["formality"],
              "request": "smoke test", "title": outfit["title"]}
    call("POST", "/feedback", {**common, "vote": "up"})
    call("POST", "/wears", {**common, "worn": True})
    _, hist = call("GET", "/history")
    assert any(set(v["item_ids"]) == set(ids) for v in hist["votes"]), "vote not saved"
    call("POST", "/wears", {**common, "worn": False})       # undo both again
    call("POST", "/feedback", {**common, "vote": None})
    return "saved a 👍 and a wear, then removed them"


print(f"Checking {BASE} ...\n")
if not check("Website pages load", website):
    print("\nIs the server running?  uvicorn outfit_engine:app --reload")
    sys.exit(1)
if check("Items list (GET /items)", items):
    check("One item (GET /items/{id})", one_item)
    check("Thumbnails (GET /thumb/...)", thumbnails)
check("Tag lists (GET /allowed)", allowed)
check("Feedback tables (GET /history)", history)
if check("Outfits (POST /outfits, uses OpenAI)", outfits):
    check("Swap one piece (POST /outfits/swap, uses OpenAI)", swap)
    if WRITE:
        check("Feedback round trip (--write)", feedback_round_trip)
print(f"\n{sum(results)} of {len(results)} checks passed.")
if not all(results):
    print("If 'Feedback tables' failed: run  psql -d wardrobe -f add_feedback.sql")
