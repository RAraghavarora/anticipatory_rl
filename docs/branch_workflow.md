# Branch workflow for experiment isolation

## Branch ownership
- `main`: shared framework/core only.
- `restaurant`: restaurant experiment code and scripts (`paper_restaurant/**` plus restaurant-specific env/config/agents).
- `blocksworld`: blocksworld experiment code (`paper1_blockworld/**` plus blocksworld-specific wiring).
- `misc`: temporary or unrelated experiments.

## Current topology
- Canonical restaurant branch is `restaurant` (renamed from `restaurant_report`).
- `blocksworld` and `misc` are based on `main`.
- Recommended worktrees:
  - `../wt-main`
  - current repo checkout as `restaurant`
  - `../wt-blocksworld`
  - `../wt-misc`

Use:

```bash
./scripts/setup_experiment_worktrees.sh
```

## Keeping branches in sync
Rebase each long-lived branch onto `main` before starting new branch-specific work:

```bash
./scripts/sync_experiment_branches.sh
```

Resolve conflicts immediately and commit on the branch that owns the path.

## Conflict prevention rules
- Avoid editing the same file in multiple experiment branches.
- Keep experiment launch/inference scripts under branch-owned directories.
- Keep generated outputs out of Git (`runs/`, slurm logs, latex aux files).
- Final integration order: `restaurant` -> `blocksworld` -> `misc` into `main`.
