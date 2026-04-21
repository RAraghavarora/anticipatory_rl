#!/usr/bin/env python3
from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "paper_restaurant" / "scripts" / "restaurant_multi_seed_infer.py"
    runpy.run_path(str(target), run_name="__main__")
