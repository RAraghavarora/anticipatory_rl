# Thor/ProcTHOR Environment Setup

The new `ThorRearrangementEnv` lets us run the same pick-and-place style task inside AI2-THOR or ProcTHOR scenes. This guide assumes you are working inside the existing `thesis` Conda environment.

## 1. Environment preparation

```bash
conda activate thesis
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first `Controller` instantiation will download the Unity builds (~500 MB) into `~/.ai2thor`. Make sure you have that much free space.

## 2. Smoke test (built-in iTHOR scenes)

```bash
python scripts/test_thor_env.py --scene FloorPlan1 --episodes 2 --max-steps 150
```

The script:

1. spins up `ThorRearrangementEnv` with the specified scene pool,
2. samples a random pickupable object and receptacle target,
3. rolls random actions while logging rewards + success flags.

You should see console prints such as:

```
Episode 00 scene=FloorPlan1 task=Bottle -> CounterTop
  step=000 action=0 reward=-0.01 success=False horizon=False err=None
...
```

Stop with `Ctrl+C`.

## 3. Loading ProcTHOR houses

### 3.1. Existing JSON houses

If you already have ProcTHOR JSONs on disk (for example under `data/procthor/*.json`), point the demo script to them:

```bash
python scripts/test_thor_env.py --house-glob 'data/procthor/*.json' --episodes 1
```

The wrapper automatically deserializes each JSON and calls `controller.reset(scene="Procedural", house=house_json)`.

### 3.2. Generating houses on the fly

You can hook any callable that returns a ProcTHOR-style house dictionary. For instance, with the upstream generator:

```python
from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
from anticipatory_rl.envs.thor_rearrangement_env import ThorRearrangementEnv

house_gen = HouseGenerator(split="train", seed=0, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER)

def sample_house(rng):
    house, _ = house_gen.sample()
    return house.to_dict()

env = ThorRearrangementEnv(procthor_house_sampler=sample_house)
```

This feeds freshly generated layouts into the environment each time you call `reset`.

## 4. Training integration

`ThorRearrangementEnv` mirrors the custom `(obs, reward, success, horizon, info)` API used by `SimpleGridImageEnv`. That means you can swap it into the PPO/DQN trainers with minimal plumbing:

```python
env = ThorRearrangementEnv(scene_pool=[\"FloorPlan1\", \"FloorPlan2\"], max_task_steps=256)
obs, info = env.reset()
```

Wrap it with the existing `SimpleGridWrapper` (or a similar Gym wrapper) if you want SB3-compatible `(terminated, truncated)` outputs.

## 5. Useful CLI knobs

* `--object-types` & `--receptacle-types`: limit task sampling to the objects relevant for your experiments.
* `--width/--height`: control the observation resolution (default 400×300).
* `--house-json/--house-glob`: use ProcTHOR assets instead of stock iTHOR layouts.

Refer back to `scripts/test_thor_env.py` for a complete list of switches.
