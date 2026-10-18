#!/bin/bash
# Moves the files you chose to remove into _to_delete/ instead of deleting them.
# Check _to_delete/ in Finder, then drag the whole folder to the Bin.
# (They also stay in your git history, so you can always get them back.)
cd "$(dirname "$0")" || exit 1
mkdir -p _to_delete/test_images

move() { if [ -e "$1" ]; then mv "$1" "$2" && echo "moved  $1"; fi; }

# BiRefNet: keep the script (a Linux-friendly cut-out option for putting the app online);
# move its 60 old cut-outs (120 MB) and its downloaded tools (re-downloaded on first run)
mkdir -p _to_delete/birefnet_generate
for f in birefnet_generate/*.png; do move "$f" _to_delete/birefnet_generate/; done
move birefnet_generate/bin        _to_delete/birefnet_generate/
for f in test_images/failed_*.jpg; do move "$f" _to_delete/test_images/; done
move model_consistency_test.py    _to_delete/      # one-off model comparison
move model_runs.csv               _to_delete/      # its output, if it exists
move export_view_items_2.py       _to_delete/      # table export for your mentor
move view_items_2.csv             _to_delete/      # its output

echo "Done. Check _to_delete/, then drag it to the Bin."
