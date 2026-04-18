#!/bin/bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
repo_parent="$(dirname "$repo_root")"

declare -A branch_paths=(
  [restaurant]="${repo_root}"
  [blocksworld]="${repo_parent}/wt-blocksworld"
  [misc]="${repo_parent}/wt-misc"
)

for branch in restaurant blocksworld misc; do
  path="${branch_paths[$branch]}"
  if [ ! -e "$path/.git" ]; then
    echo "Worktree missing for ${branch}: ${path}" >&2
    exit 1
  fi
  git -C "$path" checkout "$branch" >/dev/null
  git -C "$path" rebase main
done
