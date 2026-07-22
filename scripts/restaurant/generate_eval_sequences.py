import json
from pathlib import Path
from anticipatory_rl.envs.restaurant.env import RestaurantSymbolicEnv

def generate_sequences(num_sequences: int = 10, tasks_per_seq: int = 50, base_seed: int = 42):
    out_dir = Path("experiments/sequences")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_sequences):
        env = RestaurantSymbolicEnv(config_path="configs/restaurant/toy_level_3.yaml", rng_seed=base_seed+i)
        env.reset(seed=base_seed+i)
        
        tasks = []
        for _ in range(tasks_per_seq):
            # Advance the environment to get a new IID task properly
            # _resample_task generates the next task and returns it
            env._resample_task()
            task = env.task
            
            # Convert RestaurantTask to dict, removing None values to keep JSON clean
            task_dict = {"task_type": task.task_type}
            if task.target_location is not None:
                task_dict["target_location"] = task.target_location
            if task.target_kind is not None:
                task_dict["target_kind"] = task.target_kind
            if task.object_name is not None:
                task_dict["object_name"] = task.object_name
                
            tasks.append(task_dict)
            
        seq_id = f"iid-eval-seq-{i:02d}"
        seq_data = {
            "sequence_id": seq_id,
            "tasks": tasks
        }
        
        out_path = out_dir / f"{seq_id}.json"
        out_path.write_text(json.dumps(seq_data, indent=2) + "\n", encoding="utf-8")
        print(f"Generated {out_path} ({tasks_per_seq} tasks)")

if __name__ == "__main__":
    generate_sequences()
