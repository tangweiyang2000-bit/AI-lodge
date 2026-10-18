#!/bin/bash
# Makes `main` an exact copy of this version (branch data-prep-v2), on your Mac and on GitHub.
#   1. saves (commits) and pushes this version
#   2. backs up the old main as the tag `old-main-backup` (so nothing is truly lost)
#   3. points main at this version and pushes it
# It stops at the first problem and explains it. Run from anywhere:  bash make_main.sh
set -u
V2="$HOME/Desktop/AI wardobe-data-prep-v2"
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

echo "== 2. Finding main"
git fetch -q origin || fail "couldn't reach GitHub"
MB=main
git rev-parse -q --verify "origin/main" >/dev/null || git rev-parse -q --verify "main" >/dev/null || MB=master
# Is main open in one of your folders? (then that folder must be updated too)
OPEN_IN=$(git worktree list --porcelain | awk -v b="branch refs/heads/$MB" '/^worktree /{w=substr($0,10)} $0==b{print w}')

echo "== 3. Backing up the old $MB as tag old-main-backup"
if git rev-parse -q --verify "origin/$MB" >/dev/null; then OLD="origin/$MB"; else OLD="$MB"; fi
git tag -f old-main-backup "$OLD" >/dev/null || fail "couldn't create the backup tag"
git push -f origin old-main-backup || fail "couldn't upload the backup tag"

echo "== 4. Making $MB an exact copy of data-prep-v2"
if [ -n "$OPEN_IN" ]; then
  cd "$OPEN_IN" || fail "can't open $OPEN_IN"
  [ -z "$(git status --porcelain)" ] || { git status --short; fail "$OPEN_IN has unsaved changes (listed above). Old main is safe in tag old-main-backup."; }
  git reset -q --hard data-prep-v2 || fail "reset failed (old main is safe in tag old-main-backup)"
else
  git branch -f "$MB" data-prep-v2 || fail "couldn't move $MB (old main is safe in tag old-main-backup)"
fi
git push --force-with-lease origin "$MB" || fail "couldn't upload $MB (old main is safe in tag old-main-backup)"

echo
echo "DONE: $MB now matches data-prep-v2, on your Mac and on GitHub."
echo "Old version kept as tag: old-main-backup"
git log --oneline -3 "$MB"
