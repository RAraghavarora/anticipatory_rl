from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Minimal set of named colors used by the MiniWorldGridRearrange environment.
COLORS = {
    "red": np.array([1.0, 0.0, 0.0]),
    "green": np.array([0.0, 0.7, 0.2]),
    "blue": np.array([0.0, 0.45, 0.85]),
    "yellow": np.array([1.0, 0.85, 0.2]),
    "purple": np.array([0.6, 0.2, 0.8]),
    "orange": np.array([1.0, 0.6, 0.0]),
    "teal": np.array([0.0, 0.65, 0.65]),
    "pink": np.array([1.0, 0.45, 0.7]),
    "navy": np.array([0.1, 0.15, 0.45]),
    "grey": np.array([0.6, 0.6, 0.6]),
    "white": np.array([1.0, 1.0, 1.0]),
    "black": np.array([0.0, 0.0, 0.0]),
    "brown": np.array([0.55, 0.27, 0.07]),
}


@dataclass
class Box:
    color: str
    size: Sequence[float] | float
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    dir: float = 0.0

    def __post_init__(self) -> None:
        self.size = self._normalize_size(self.size)

    @staticmethod
    def _normalize_size(size: Sequence[float] | float) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(size, dtype=float))
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        return arr
