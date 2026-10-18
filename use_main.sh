#!/bin/bash
# Switches this folder to `main` and removes the branches data-prep-v1 and data-prep-v2
# (on your Mac and on GitHub). Both are first saved as tags, so nothing is truly lost:
#   old-v1-backup, old-v2-backup
# Run after make_main.sh:  bash use_main.sh
set -u
V2="$HOME/Desktop/AI wardobe-data-prep-v2"
V1="$HOME/Desktop/AI wardobe-data-prep-v1"
fail() { echo; echo "STOPPED: $1"; exit 1; }

cd "$V2" || fail "can't find $V2"
git fetch -q origin || fail "couldn't reach GitHub"

echo "== 1. Checking main is up to date"
git rev-parse -q --verify main >/dev/null || fail "there is no main yet. Run: bash make_main.sh"
if git rev-parse -q --verify data-prep-v2 >/dev/null; then
  [ "$(git rev-parse main)" = "$(git rev-parse data-prep-v2)" ] || fail "main is not a copy of data-prep-v2 yet. Run: bash make_main.sh"
fi
[ "$(git rev-parse main)" = "$(git rev-parse origin/main)" ] || fail "main on GitHub is different. Run: bash make_main.sh"

echo "== 2. Switching this folder to main"
git switch -q main || fail "couldn't switch to main"
git branch -q -u origin/main main

echo "== 3. Saving the old branches as tags"
for b in data-prep-v1 data-prep-v2; do
  tag="old-${b#data-prep-}-backup"
  if git rev-parse -q --verify "origin/$b" >/dev/null; then src="origin/$b"
  elif git rev-parse -q --verify "$b" >/dev/null; then src="$b"
  else continue; fi
  git tag -f "$tag" "$src" >/dev/null && git push -q -f origin "$tag" || fail "couldn't save $b as tag $tag"
  echo "   $b -> tag $tag"
done

echo "== 4. Removing data-prep-v1 and data-prep-v2"
if [ -d "$V1" ] && [ "$(git -C "$V1" branch --show-current)" = "data-prep-v1" ]; then
  git -C "$V1" switch -q --detach || fail "couldn't free data-prep-v1 in the v1 folder"
fi
for b in data-prep-v1 data-prep-v2; do
  git rev-parse -q --verify "$b" >/dev/null && { git branch -q -D "$b" || fail "couldn't remove $b"; }
  git rev-parse -q --verify "origin/$b" >/dev/null && { git push -q origin --delete "$b" || fail "couldn't remove $b from GitHub"; }
  echo "   removed $b"
done

echo
echo "DONE: this folder is on main. Branches left:"
git branch -a
