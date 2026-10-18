#!/bin/bash
# Turns this folder into a normal, standalone git project (no longer tied to the v1 folder)
# and renames it to "AI wardrobe" on your Desktop. Your files, .env, .venv and uploads stay.
# Run after use_main.sh:  bash standalone.sh
set -u
OLD="$HOME/Desktop/AI wardobe-data-prep-v2"
NEW="$HOME/Desktop/AI wardrobe"
fail() { echo; echo "STOPPED: $1"; exit 1; }

echo "== 1. Checking everything is saved on GitHub"
cd "$OLD" || fail "can't find $OLD"
[ -e "$NEW" ] && fail "$NEW already exists. Rename or remove it first."
[ "$(git branch --show-current)" = "main" ] || fail "this folder isn't on main yet. Run: bash make_main.sh  then  bash use_main.sh"
git fetch -q origin || fail "couldn't reach GitHub"
[ -z "$(git status --porcelain)" ] || { git status --short; fail "there are unsaved changes (listed above). Save them first: git add -A && git commit -m \"Save\" && git push"; }
[ "$(git rev-parse main)" = "$(git rev-parse origin/main)" ] || fail "main isn't fully uploaded. Run: git push"
[ -z "$(git stash list)" ] || fail "you have stashed changes (git stash list). Apply or drop them first."
URL=$(git remote get-url origin)

echo "== 2. Giving this folder its own git history"
[ -f .git ] || fail ".git here is already a normal folder, so this is already standalone"
LINK="$(mktemp -d)/old-git-link" && mv .git "$LINK" || fail "couldn't move the old .git link"
git init -q -b main || { mv "$LINK" .git; fail "git init failed (put things back)"; }
git remote add origin "$URL"
if ! git fetch -q --tags origin; then
  rm -rf .git; mv "$LINK" .git; fail "couldn't download from GitHub (put things back)"
fi
git reset -q origin/main          # files untouched, git now knows they are the latest main
git branch -q -u origin/main main
[ -z "$(git status --porcelain)" ] || { git status --short; rm -rf .git; mv "$LINK" .git; fail "files differ from GitHub (listed above), so nothing was changed. Paste this to Claude."; }

echo "== 3. Renaming the folder to: AI wardrobe"
cd "$HOME/Desktop" && mv "$OLD" "$NEW" || fail "couldn't rename the folder (is something using it?)"
# .venv remembers its folder path, so update it
if [ -d "$NEW/.venv" ]; then
  grep -rlIF "$OLD" "$NEW/.venv/bin" "$NEW/.venv/pyvenv.cfg" 2>/dev/null | while IFS= read -r f; do
    O="$OLD" N="$NEW" perl -pi -e 's/\Q$ENV{O}\E/$ENV{N}/g' "$f"
  done
  "$NEW/.venv/bin/python" -c "import fastapi" 2>/dev/null && echo "   .venv works in the new folder" \
    || echo "   WARNING: .venv check failed; tell Claude"
fi

echo
echo "DONE: your project is now  ~/Desktop/AI wardrobe  (on main, linked to GitHub)."
echo "Next: in VS Code choose File > Open Folder... > Desktop > AI wardrobe"
echo "Then you can drag these old folders to the Bin:"
echo "   AI wardobe-data-prep-v1,  AI wardobe-main"
