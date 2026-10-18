#!/bin/bash
# Makes `main` an exact copy of this version (branch data-prep-v2), on your Mac and on GitHub.
#   1. saves (commits) and pushes this version
#   2. backs up the old main as the tag `old-main-backup` (so nothing is truly lost)
#   3. resets main to this version and pushes it
# It stops at the first problem and explains it. Run from anywhere:  bash make_main.sh
set -u
V2="$HOME/Desktop/AI wardobe-data-prep-v2"
V1="$HOME/Desktop/AI wardobe-data-prep-v1"
fail() { echo; echo "STOPPED: $1"; exit 1; }

echo "== 1. Saving this version (data-prep-v2)"
cd "$V2" || fail "can't find $V2"
BR=$(git branch --show-current)
[ "$BR" = "data-prep-v2" ] || fail "this folder is on branch '$BR', expected data-prep-v2"
git rm --cached -q outfits.html 2>/dev/null   # rewritten on every request, now git-ignored
git add -A
if git diff --cached --name-only | grep -qx ".env"; then
  git reset -q
  fail ".env (your API keys) was about to be saved, so nothing was committed"
fi
if git diff --cached --quiet; then
  echo "   nothing new to save"
else
  git commit -q -m "Website, feedback loop, swap, thumbnails, docs and cleanup" || fail "commit failed"
  echo "   saved: $(git log --oneline -1)"
fi
git push -u origin data-prep-v2 || fail "couldn't upload data-prep-v2 to GitHub"

echo "== 2. Checking main (in the v1 folder)"
cd "$V1" || fail "can't find $V1"
MB=$(git branch --show-current)
[ "$MB" = "main" ] || [ "$MB" = "master" ] || fail "the v1 folder is on branch '$MB', not main"
if [ -n "$(git status --porcelain)" ]; then
  git status --short
  fail "the v1 folder has unsaved changes (listed above). Nothing was changed."
fi

echo "== 3. Backing up the old $MB as tag old-main-backup"
git tag -f old-main-backup "$MB" >/dev/null || fail "couldn't create the backup tag"
git push -f origin old-main-backup || fail "couldn't upload the backup tag"

echo "== 4. Making $MB an exact copy of data-prep-v2"
git fetch -q origin
git reset --hard data-prep-v2 || fail "reset failed (old main is safe in tag old-main-backup)"
git push --force-with-lease origin "$MB" || fail "couldn't upload $MB (old main is safe in tag old-main-backup)"

echo
echo "DONE: $MB now matches data-prep-v2, on your Mac and on GitHub."
echo "Old version kept as tag: old-main-backup"
git log --oneline -3
