from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env


class MiniWorldEnv(Env):
    """Lightweight stub of the MiniWorld base class sufficient for testing."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        max_episode_steps: int = 1500,
        view: str = "top",
        render_mode: Optional[str] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.view = view
        self.render_mode = render_mode
        self.step_count = 0
        self.rooms: List[Dict[str, float]] = []
        self.entities: List[Any] = []
        self.vis_fb: Optional[np.ndarray] = None
        self._rng = np.random.default_rng()
        self.agent = SimpleNamespace(pos=np.zeros(3), dir=0.0, carrying=None)
        self.observation_space = None
        self.action_space = None

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        del options  # Unused
        if seed is not None:
            self.seed(seed)
        self.step_count = 0
        self.rooms.clear()
        self.entities.clear()
        self._gen_world()
        obs = self.render_obs()
        info: Dict[str, Any] = {}
        if hasattr(self, "get_state"):
            try:
                info["state"] = self.get_state()  # type: ignore[attr-defined]
            except Exception:
                pass
        return obs, info

    def render_obs(self):
        frame = self.render_top_view(self.vis_fb)
        self.vis_fb = frame
        return frame

    def render(self):
        return self.render_obs()

    def close(self):
        self.entities.clear()

    # ------------------------------------------------------------------ helpers used by MiniWorldGridRearrange
    def add_rect_room(
        self,
        *,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        **kwargs: Any,
    ) -> None:
        self.rooms.append(
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_z": min_z,
                "max_z": max_z,
                **kwargs,
            }
        )

    def place_entity(self, entity: Any, pos: np.ndarray, dir: float = 0.0):
        entity.pos = np.asarray(pos, dtype=float)
        entity.dir = dir
        self.entities.append(entity)
        return entity

    def place_agent(self, pos: np.ndarray, dir: float = 0.0):
        self.agent.pos = np.asarray(pos, dtype=float)
        self.agent.dir = dir
        self.agent.carrying = None

    def _get_carry_pos(self, agent_pos: np.ndarray, entity: Any) -> np.ndarray:
        del entity  # Position does not depend on entity size in the stub.
        offset = np.array([0.0, 0.5, 0.0])
        return np.asarray(agent_pos, dtype=float) + offset

    # ------------------------------------------------------------------ abstract hooks
    def _gen_world(self) -> None:
        raise NotImplementedError

    def render_top_view(self, frame_buffer: Optional[np.ndarray]):
        raise NotImplementedError
