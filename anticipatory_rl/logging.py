"""Experiment logging for restaurant RL training."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

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


class WandbLogger:
    def __init__(self, args: argparse.Namespace, run_label: str) -> None:
        self._run = None
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[train] W&B logging disabled: install `wandb` to enable experiment tracking.")
            return

        self._wandb = wandb
        try:
            self._run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "restaurant_rl_factored"),
                entity=os.environ.get("WANDB_ENTITY"),
                name=run_label,
                config={
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
            )
        except Exception as exc:
            print(f"[train] W&B logging disabled: wandb.init() failed ({exc!r}).")
            return
        self._defined_metrics: set[str] = set()
        self._run.config.update({"action_space": {"action_types": list(ACTION_TYPES), "factored": True}})
        print("[train] W&B logging enabled.")

    @staticmethod
    def _metric_name(name: str, context: Mapping[str, object] | None) -> str:
        if not context:
            return name
        suffix = "/".join(f"{key}={value}" for key, value in sorted(context.items()))
        return f"{name}/{suffix}"

    def set_metadata(self, key: str, value: object) -> None:
        if self._run is not None:
            self._run.config.update({key: value}, allow_val_change=True)

    def track(
        self,
        value: float | int,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self._run is not None:
            metric_name = self._metric_name(name, context)
            step_name = f"{metric_name}_step"
            if metric_name not in self._defined_metrics:
                self._run.define_metric(step_name, hidden=True)
                self._run.define_metric(metric_name, step_metric=step_name)
                self._defined_metrics.add(metric_name)
            self._run.log({metric_name: value, step_name: step})

    def track_text(self, text: str, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        if self._run is not None:
            self._run.log({self._metric_name(name, context): self._wandb.Html(f"<pre>{text}</pre>")})

    def track_image(self, image_path: Path, *, name: str, step: int, context: Mapping[str, object] | None = None) -> None:
        if self._run is not None:
            self._run.log({self._metric_name(name, context): self._wandb.Image(str(image_path))})

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class CSVLogger:
    """Lightweight local CSV logger mirroring the AimLogger interface.

    Writes numeric metrics in long format to ``runs/<run_label>/metrics.csv``
    and metadata to ``runs/<run_label>/metadata.json``. Each ``track`` call
    produces one row with a fixed column schema, so new metrics can appear
    at any time without corrupting the file. Text and image tracks are ignored
    for CSV but are still suitable for Aim.

    To get a wide-format DataFrame::

        df = pd.read_csv("runs/<run_label>/metrics.csv")
        df_wide = df.pivot_table(
            index=["run_label", "step"],
            columns="metric",
            values="value",
        ).reset_index()
    """

    def __init__(
        self,
        args: argparse.Namespace,
        run_label: str,
        run_dir: Path | None = None,
        flush_every: int = 1000,
    ) -> None:
        self.run_label = run_label
        self.run_dir = Path(run_dir) if run_dir is not None else Path("runs") / run_label
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.metadata_path = self.run_dir / "metadata.json"
        self.flush_every = max(1, flush_every)

        self._rows: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._tracks_since_flush = 0
        self._closed = False
        self._session_started = False

        # Truncate any stale metrics.csv from a prior run reusing this label,
        # so early inspection/crash-before-flush never reads old data.
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=["run_label", "step", "metric", "context", "value"]).writeheader()
        self._session_started = True

        # Write hparams as metadata immediately.
        self._metadata["hparams"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        self._metadata["action_space"] = {"action_types": list(ACTION_TYPES), "factored": True}
        self._write_metadata()

        print(f"[train] CSV logging enabled -> {self.csv_path}")

    @staticmethod
    def _metric_name(name: str, context: Mapping[str, object] | None) -> str:
        if not context:
            return name
        suffix = "__".join(f"{key}_{value}" for key, value in sorted(context.items()))
        return f"{name}__{suffix}"

    @staticmethod
    def _context_str(context: Mapping[str, object] | None) -> str:
        if not context:
            return ""
        return "__".join(f"{key}_{value}" for key, value in sorted(context.items()))

    def _write_metadata(self) -> None:
        try:
            with self.metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2, default=str)
        except Exception as exc:
            print(f"[CSVLogger] Warning: failed to write metadata: {exc}")

    def set_metadata(self, key: str, value: object) -> None:
        self._metadata[key] = value
        self._write_metadata()

    def track(
        self,
        value: float | int,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self._closed:
            return
        self._rows.append({
            "run_label": self.run_label,
            "step": step,
            "metric": self._metric_name(name, context),
            "context": self._context_str(context),
            "value": value,
        })
        self._tracks_since_flush += 1
        if self._tracks_since_flush >= self.flush_every:
            self._flush()

    def track_text(
        self,
        text: str,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        # Text traces are intentionally Aim-only; skip for CSV.
        return

    def track_image(
        self,
        image_path: Path,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        # Image traces are intentionally Aim-only; skip for CSV.
        return

    def _flush(self) -> None:
        if not self._rows:
            return
        try:
            # First flush of a fresh session truncates the file so reusing a
            # run_label across runs does not stack metrics from prior runs.
            mode = "w" if not self._session_started else "a"
            with self.csv_path.open(mode, newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["run_label", "step", "metric", "context", "value"])
                if mode == "w":
                    writer.writeheader()
                writer.writerows(self._rows)
            self._session_started = True
        except Exception as exc:
            print(f"[CSVLogger] Warning: failed to flush metrics: {exc}")
        finally:
            self._rows.clear()
            self._tracks_since_flush = 0

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._closed = True


class LoggerPair:
    """Dispatch calls to configured experiment loggers with one interface."""

    def __init__(
        self,
        aim_logger: AimLogger | None = None,
        csv_logger: CSVLogger | None = None,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        self.aim = aim_logger
        self.csv = csv_logger
        self.wandb = wandb_logger

    def set_metadata(self, key: str, value: object) -> None:
        if self.aim is not None:
            self.aim.set_metadata(key, value)
        if self.csv is not None:
            self.csv.set_metadata(key, value)
        if self.wandb is not None:
            self.wandb.set_metadata(key, value)

    def track(
        self,
        value: float | int,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self.aim is not None:
            self.aim.track(value, name=name, step=step, context=context)
        if self.csv is not None:
            self.csv.track(value, name=name, step=step, context=context)
        if self.wandb is not None:
            self.wandb.track(value, name=name, step=step, context=context)

    def track_text(
        self,
        text: str,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self.aim is not None:
            self.aim.track_text(text, name=name, step=step, context=context)
        if self.wandb is not None:
            self.wandb.track_text(text, name=name, step=step, context=context)

    def track_image(
        self,
        image_path: Path,
        *,
        name: str,
        step: int,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self.aim is not None:
            self.aim.track_image(image_path, name=name, step=step, context=context)
        if self.wandb is not None:
            self.wandb.track_image(image_path, name=name, step=step, context=context)

    def close(self) -> None:
        if self.aim is not None:
            self.aim.close()
        if self.csv is not None:
            self.csv.close()
        if self.wandb is not None:
            self.wandb.close()
