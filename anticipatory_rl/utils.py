"""Shared utilities for restaurant RL training and inference."""

from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np
import torch


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_masks(info: Mapping[str, np.ndarray | List[float] | Dict[str, object]]) -> Dict[str, np.ndarray]:
    return {
        "valid_action_type_mask": np.asarray(info.get("valid_action_type_mask"), dtype=np.float32),
        "valid_object1_mask": np.asarray(info.get("valid_object1_mask"), dtype=np.float32),
        "valid_location_mask": np.asarray(info.get("valid_location_mask"), dtype=np.float32),
        "valid_object2_mask": np.asarray(info.get("valid_object2_mask"), dtype=np.float32),
    }


def masked_choice(values: torch.Tensor, mask: torch.Tensor) -> int:
    valid = torch.nonzero(mask > 0.0, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return int(torch.argmax(values).item())
    masked = values.clone()
    masked[mask <= 0.0] = float("-inf")
    return int(torch.argmax(masked).item())


def random_valid_index(mask: np.ndarray) -> int:
    indices = np.flatnonzero(mask > 0.0)
    if indices.size == 0:
        return int(mask.shape[0] - 1)
    return int(np.random.choice(indices.tolist()))
