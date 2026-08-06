import json, glob, os, sys
base = sys.argv[1] if len(sys.argv) > 1 else "runs/v5eval"
rows = []
for f in sorted(glob.glob(os.path.join(base, "base_*.json"))):
    try: d = json.load(open(f))
    except Exception: continue
    rows.append((os.path.basename(f), d.get("K"), d.get("whole_sequence_physical_cost"),
                 d.get("sequence_complete"), d.get("valid_prefix_completion_count")))
print(f"{'file':30s} {'K':>2} {'cost':>10} {'done':>5} {'n/50':>5}")
for r in rows:
    print(f"{r[0]:30s} {str(r[1]):>2} {str(r[2]):>10} {str(r[3]):>5} {str(r[4]):>5}")
g = sorted(glob.glob(os.path.join(base, "guided_*.json")))
print(f"\nguided files: {len(g)}, non-empty: {sum(1 for x in g if os.path.getsize(x)>0)}")
for f in g:
    if os.path.getsize(f) == 0: continue
    try: d = json.load(open(f))
    except Exception: continue
    m, gu = d.get("myopic", {}).get("summary", {}), d.get("guided", {}).get("summary", {})
    print(f"  {os.path.basename(f):34s} myopic={m.get('total_pddl_cost')} "
          f"guided={gu.get('total_pddl_cost')} delta={d.get('paired_cost_delta')} "
          f"jar_tasks={gu.get('jar_ready_task_count')}")
