"""Aim logging for restaurant RL training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from anticipatory_rl.envs.restaurant.env import ACTION_TYPES


class AimLogger:
    def __init__(self, args: argparse.Namespace, run_label: str) -> None:
        self._run = None
        try:
            from aim import Run  # type: ignore
        except ImportError:
            print("[train] Aim logging disabled: install `aim` to enable experiment tracking.")
            return

        self._run = Run(experiment="restaurant_rl_factored")
        self._run["run_label"] = run_label
        self._run["hparams"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        self._run["action_space"] = {"action_types": list(ACTION_TYPES), "factored": True}
        print("[train] Aim logging enabled. Launch UI with `aim up`.")

    def set_metadata(self, key: str, value: object) -> None:
        if self._run is not None:
            self._run[key] = value

    def track(
        self,
        value: float | int,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self._run is None:
            return
        self._run.track(value, name=name, step=step, context=dict(context or {}))

    def close(self) -> None:
        if self._run is None:
            return
        close = getattr(self._run, "close", None)
        if callable(close):
            close()

    def track_text(self, text: str, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        if self._run is None:
            return
        try:
            from aim import Text  # type: ignore
            self._run.track(Text(text), name=name, step=step, context=dict(context or {}))
        except Exception:
            self._run.track(text, name=name, step=step, context=dict(context or {}))

    def track_image(self, image_path: Path, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        if self._run is None:
            return
        try:
            from aim import Image  # type: ignore
            self._run.track(Image(str(image_path)), name=name, step=step, context=dict(context or {}))
        except Exception:
            self._run.track(str(image_path), name=name, step=step, context=dict(context or {}))
