"""Compare anticipatory vs myopic plans side-by-side. Find divergences."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANT = json.loads((ROOT / "runs/v3/infer_compare/anticipatory/plans.json").read_text())
MYO = json.loads((ROOT / "runs/v3/infer_compare/myopic/plans.json").read_text())

print(f"anticipatory plans: {len(ANT)}, myopic plans: {len(MYO)}")
print()

# Both should see the same task sequence (same seed). Compare task-by-task.
n = min(len(ANT), len(MYO))
print(f"{'#':>3} {'task_type':<18} {'tgt':<14} {'ant_succ':<8} {'myo_succ':<8} {'ant_steps':<9} {'myo_steps':<9} {'same_plan'}")
print("-" * 95)
diverge_count = 0
for i in range(n):
    a, m = ANT[i], MYO[i]
    same = a["actions"] == m["actions"]
    if not same:
        diverge_count += 1
    tt = a["task_type"] or "?"
    tgt = a.get("target_location") or a.get("target_kind") or "-"
    print(f"{i+1:>3} {tt:<18} {tgt:<14} {str(a['success']):<8} {str(m['success']):<8} {a['steps']:<9} {m['steps']:<9} {same}")

print(f"\nDiverged plans: {diverge_count}/{n}")

# Show a few diverged plans in detail
print("\n" + "=" * 80)
print("DETAILED DIVERGENCES (first 8)")
print("=" * 80)
shown = 0
for i in range(n):
    a, m = ANT[i], MYO[i]
    if a["actions"] == m["actions"]:
        continue
    shown += 1
    tt = a["task_type"] or "?"
    tgt = a.get("target_location") or a.get("target_kind") or "-"
    print(f"\n--- Task {i+1}: {tt} (target={tgt}) ---")
    print(f"  ANT: succ={a['success']} steps={a['steps']} ret={a['return']:.1f} auto={a['auto_satisfied']}")
    print(f"  MYO: succ={m['success']} steps={m['steps']} ret={m['return']:.1f} auto={m['auto_satisfied']}")
    print(f"  ANT plan ({len(a['actions'])} actions):")
    for j, act in enumerate(a["actions"]):
        print(f"    {j+1:>2}. {act}")
    print(f"  MYO plan ({len(m['actions'])} actions):")
    for j, act in enumerate(m["actions"]):
        print(f"    {j+1:>2}. {act}")
    if shown >= 8:
        break
