from kstar_planner import planners
from pathlib import Path

domain_file = Path("toy_restaurant_domain.pddl")
problem_file = Path("toy_restaurant_problem.pddl")

heuristic = "ipdb(transform=undo_to_origin())"

plans = planners.plan_topq(
    domain_file=domain_file,
    problem_file=problem_file,
    quality_bound=10.0,
    number_of_plans_bound=100,
    timeout=30,
    search_heuristic=heuristic,
)

# Write returned plans to disk in Fast Downward plan-file format
output_dir = Path("found_plans")
output_dir.mkdir(exist_ok=True)

for i, plan in enumerate(plans.get("plans", []), start=1):
    plan_file = output_dir / f"plan_{i:04d}.pddl"
    plan_file.write_text("\n".join(plan["actions"]) + "\n")
    print(f"Wrote {plan_file}  (cost={plan['cost']}, actions={len(plan['actions'])})")

print(f"\nTotal plans returned: {len(plans.get('plans', []))}")
print(f"Unsolvable: {plans.get('unsolvable', False)}")
print(f"Timeout: {plans.get('timeout_triggered', False)}")
