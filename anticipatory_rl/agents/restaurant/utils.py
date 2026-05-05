import torch
import argparse
import numpy as np

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception as exc:  # pragma: no cover - depends on runtime CUDA state
            raise RuntimeError(
                "CUDA is available but failed to select cuda:0. "
                "Check CUDA_VISIBLE_DEVICES and launcher-provided GPU rank env vars."
            ) from exc
        return torch.device("cuda:0")
    return torch.device("cpu")

def epsilon_by_step(step: int, start: float, final: float, decay: int) -> float:
    if decay <= 0:
        return final
    return final + (start - final) * np.exp(-float(step) / float(decay))

def resolve_run_label(args: argparse.Namespace) -> str:
    if args.run_label is not None:
        return args.run_label
    return "myopic_restaurant" if args.tasks_per_episode <= 1 else "anticipatory_restaurant"
