"""Tests for task 2 of gnn-steelman-jar-augmentation: threading
`--unbounded-jar-augmentation` through scripts/gnn/generate_data_aug.py.

Only exercises argparse defaults and metadata construction directly — never
generates a real dataset (that requires Fast Downward and is slow).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_GNN_DIR = _REPO_ROOT / "scripts" / "gnn"
for _p in (_REPO_ROOT, _SCRIPT_GNN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from generate_data_aug import _build_arg_parser, _build_metadata  # noqa: E402

_CONFIG_PATH = _REPO_ROOT / "configs" / "restaurant" / "toy_level_3.yaml"

_REQUIRED_ARGV = [
    "--config-path", str(_CONFIG_PATH),
    "--planner-path", "downward/fast-downward.py",
    "--domain-path", "pddl/toy_restaurant_domain.pddl",
    "--output-path", "runs/unused.pt",
]


def test_flag_default_is_false():
    args = _build_arg_parser().parse_args(_REQUIRED_ARGV)
    assert args.unbounded_jar_augmentation is False


def test_metadata_records_flag_default_false():
    args = _build_arg_parser().parse_args(_REQUIRED_ARGV)
    metadata = _build_metadata(args)
    assert metadata["unbounded_jar_augmentation"] is False
    # Provenance keys the brief calls out, matching the naming style used by
    # the other demo-generation metadata blocks in this repo.
    for key in ("config_hash", "max_augs", "seed", "unbounded_jar_augmentation"):
        assert key in metadata


def test_metadata_records_flag_when_set():
    args = _build_arg_parser().parse_args(_REQUIRED_ARGV + ["--unbounded-jar-augmentation"])
    assert args.unbounded_jar_augmentation is True
    metadata = _build_metadata(args)
    assert metadata["unbounded_jar_augmentation"] is True
