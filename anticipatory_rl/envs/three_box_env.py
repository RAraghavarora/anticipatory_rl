"""Three-Box toy environment for testing anticipatory RL.

Grid layout (5x5):
    . . C . .     C = Receptacle C at (2,0) — apple spawn
    . . . . .
    . . . . .
    . . . . .
    A . . . B     A = (0,4), B = (4,4) — target receptacles

Distances (Manhattan): C→A = C→B = 6 (symmetric), A→B = 4.

Each meta-episode has exactly two sequential tasks:

  Task 1 — Clear C
      The apple spawns on C.  The agent must pick it up and place it
      on A or B.  Placing back on C or on the floor is not allowed.

  Task 2 — Move
      Sampled once when Task 1 completes:
        prob_a  → "Move apple to A"
        1-prob_a → "Move apple to B"
      If the apple is already at the target (anticipatory placement),
      the next step auto-completes with full reward and zero step cost.

The episode terminates (terminated=True) when Task 2 is completed, or
truncates (truncated=True) when max_episode_steps is exceeded.

Observation  (6 × H × W, float32):
    Ch 0-2  RGB top-down render normalised to [0, 1]
    Ch 3    Binary mask — apple's current tile
    Ch 4    Binary mask — target receptacle's tile
    Ch 5    Scalar plane: 1.0 = clear task, 0.0 = move task
"""

from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import numpy as np
from gymnasium import Env, spaces
from PIL import Image, ImageDraw

Coord = Tuple[int, int]

GRID_SIZE = 5

REC_A: Coord = (0, 4)
REC_B: Coord = (4, 4)
REC_C: Coord = (2, 0)

RECEPTACLES: Dict[str, Coord] = {"A": REC_A, "B": REC_B, "C": REC_C}
REC_TILES: Set[Coord] = set(RECEPTACLES.values())

REC_RGB: Dict[str, Tuple[int, int, int]] = {
    "A": (50, 200, 80),
    "B": (50, 80, 220),
    "C": (220, 70, 50),
}

APPLE_RGB = (230, 50, 30)
AGENT_FILL = (180, 30, 30)
AGENT_OUTLINE = (60, 0, 0)
OBJ_OUTLINE = (10, 10, 10)
BG_RGB = (110, 207, 246)
FLOOR_RGB = (180, 180, 180)


class ThreeBoxEnv(Env):
    """5×5 grid, 3 receptacles, 1 apple.  Two-task meta-episodes."""

    metadata: Dict[str, Any] = {"render_modes": []}

    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICK = 4
    PLACE = 5

    def __init__(
        self,
        success_reward: float = 10.0,
        step_cost: float = 0.05,
        max_episode_steps: int = 200,
        prob_a: float = 0.8,
        render_tile_px: int = 12,
        render_margin_px: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = GRID_SIZE
        self.success_reward = success_reward
        self.step_cost = step_cost
        self.max_episode_steps = max_episode_steps
        self.prob_a = prob_a

        self.tile_px = max(4, int(render_tile_px))
        self.margin_px = (
            max(2, int(render_margin_px))
            if render_margin_px is not None
            else self.tile_px
        )
        self.render_size = self.grid_size * self.tile_px + 2 * self.margin_px

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, self.render_size, self.render_size),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self.agent: Coord = (2, 2)
        self.apple: Coord = REC_C
        self.carrying: bool = False
        self.task_phase: int = 1
        self.task_type: str = "clear"
        self.target_rec: str = "C"
        self._steps: int = 0
        self._t2_steps: int = 0
        self._pending_auto: bool = False

    # -------------------------------------------------------------- reset

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.apple = REC_C
        self.carrying = False
        self.agent = self._sample_free(REC_TILES)
        self.task_phase = 1
        self.task_type = "clear"
        self.target_rec = "C"
        self._steps = 0
        self._t2_steps = 0
        self._pending_auto = False
        return self._obs(), self._info()

    # -------------------------------------------------------------- step

    def step(self, action: int):
        # Auto-complete Task 2 when the apple is already at the target.
        # No step cost — this is the "free" reward for anticipation.
        if self._pending_auto:
            self._pending_auto = False
            info = self._info()
            info.update(success=True, task2_auto=True)
            return self._obs(), self.success_reward, True, False, info

        self._steps += 1
        if self.task_phase == 2:
            self._t2_steps += 1

        reward: float = -self.step_cost
        terminated = False
        truncated = False
        task1_done = False
        drop_location: str | None = None

        # --- execute action ------------------------------------------------
        if action <= self.MOVE_RIGHT:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            nx = int(np.clip(self.agent[0] + dx, 0, self.grid_size - 1))
            ny = int(np.clip(self.agent[1] + dy, 0, self.grid_size - 1))
            self.agent = (nx, ny)
            if self.carrying:
                self.apple = (nx, ny)

        elif action == self.PICK:
            if not self.carrying and self.agent == self.apple:
                self.carrying = True

        elif action == self.PLACE:
            if self.carrying and self.agent in REC_TILES:
                if not (self.task_phase == 1 and self.agent == REC_C):
                    self.carrying = False
                    self.apple = self.agent

        # --- check Task 1 completion --------------------------------------
        if (
            self.task_phase == 1
            and not self.carrying
            and self.apple != REC_C
        ):
            task1_done = True
            reward += self.success_reward

            for name, pos in RECEPTACLES.items():
                if self.apple == pos:
                    drop_location = name
                    break

            self.task_phase = 2
            self.task_type = "move"
            self._t2_steps = 0
            self.target_rec = (
                "A" if self._rng.random() < self.prob_a else "B"
            )

            if self.apple == RECEPTACLES[self.target_rec]:
                self._pending_auto = True

        # --- check Task 2 completion --------------------------------------
        elif (
            self.task_phase == 2
            and not self.carrying
            and self.apple == RECEPTACLES[self.target_rec]
        ):
            reward += self.success_reward
            terminated = True

        # --- horizon -------------------------------------------------------
        if not terminated and self._steps >= self.max_episode_steps:
            truncated = True

        info = self._info()
        info["success"] = terminated
        if task1_done:
            info["task1_done"] = True
        if drop_location is not None:
            info["drop_location"] = drop_location

        return self._obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------- observation

    def _obs(self) -> np.ndarray:
        frame = self._render_frame()
        h, w, _ = frame.shape
        obs = np.zeros((6, h, w), dtype=np.float32)
        obs[:3] = frame.transpose(2, 0, 1)
        obs[3] = self._tile_mask(self.apple, h, w)
        obs[4] = self._tile_mask(RECEPTACLES[self.target_rec], h, w)
        obs[5].fill(1.0 if self.task_type == "clear" else 0.0)
        return obs

    def _info(self) -> dict:
        return {
            "agent": self.agent,
            "apple": self.apple,
            "carrying": self.carrying,
            "task_phase": self.task_phase,
            "task_type": self.task_type,
            "target_rec": self.target_rec,
            "steps": self._steps,
            "task2_steps": self._t2_steps,
        }

    # -------------------------------------------------------------- rendering

    def _tile_bounds(self, coord: Coord) -> Tuple[int, int, int, int]:
        x, y = coord
        x0 = self.margin_px + x * self.tile_px
        y0 = self.margin_px + y * self.tile_px
        return x0, y0, x0 + self.tile_px, y0 + self.tile_px

    def _render_frame(self) -> np.ndarray:
        sz = self.render_size
        tp = self.tile_px
        mg = self.margin_px
        img = Image.new("RGB", (sz, sz), BG_RGB)
        draw = ImageDraw.Draw(img)

        draw.rectangle([mg, mg, sz - mg - 1, sz - mg - 1], fill=FLOOR_RGB)

        for name, pos in RECEPTACLES.items():
            x0, y0, x1, y1 = self._tile_bounds(pos)
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=REC_RGB[name])

        if not self.carrying:
            obj_sz = int(tp * 0.55)
            off = max(1, (tp - obj_sz) // 2)
            bx, by, _, _ = self._tile_bounds(self.apple)
            draw.rectangle(
                [bx + off, by + off, bx + off + obj_sz, by + off + obj_sz],
                fill=APPLE_RGB,
                outline=OBJ_OUTLINE,
                width=2,
            )

        ax, ay = self.agent
        cx = mg + ax * tp + tp // 2
        cy = mg + ay * tp + tp // 2
        half = tp // 2
        tri = [
            (cx - half // 2, cy - half // 2),
            (cx - half // 2, cy + half // 2),
            (cx + half // 2, cy),
        ]
        draw.polygon(tri, fill=AGENT_FILL, outline=AGENT_OUTLINE)

        if self.carrying:
            cs = max(2, tp // 3)
            draw.rectangle(
                [cx - cs // 2, cy - cs // 2, cx + cs // 2, cy + cs // 2],
                outline=APPLE_RGB,
                width=3,
            )

        return np.asarray(img, dtype=np.float32) / 255.0

    def _tile_mask(self, coord: Coord, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        x0, y0, x1, y1 = self._tile_bounds(coord)
        mask[y0:y1, x0:x1] = 1.0
        return mask

    def _sample_free(self, exclude: Set[Coord]) -> Coord:
        while True:
            c = (
                int(self._rng.integers(0, self.grid_size)),
                int(self._rng.integers(0, self.grid_size)),
            )
            if c not in exclude:
                return c
