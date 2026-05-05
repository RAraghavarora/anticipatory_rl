import torch
import argparse
import numpy as np

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
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
