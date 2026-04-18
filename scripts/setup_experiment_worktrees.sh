#!/bin/bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
repo_parent="$(dirname "$repo_root")"

declare -A worktrees=(
  [main]="${repo_parent}/wt-main"
  [restaurant]="${repo_root}"
  [blocksworld]="${repo_parent}/wt-blocksworld"
  [misc]="${repo_parent}/wt-misc"
)

if ! git show-ref --verify --quiet refs/heads/restaurant; then
  if git show-ref --verify --quiet refs/heads/restaurant_report; then
    git branch -m restaurant_report restaurant
  else
    echo "Missing restaurant branch (or restaurant_report)." >&2
    exit 1
  fi
fi

if ! git show-ref --verify --quiet refs/heads/blocksworld; then
  git branch blocksworld main
fi

if ! git show-ref --verify --quiet refs/heads/misc; then
  git branch misc main
fi

for branch in main blocksworld misc; do
  path="${worktrees[$branch]}"
  if [ ! -e "$path" ]; then
    git worktree add "$path" "$branch"
  fi
done

git worktree list
