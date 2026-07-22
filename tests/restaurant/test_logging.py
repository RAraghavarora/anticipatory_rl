import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from anticipatory_rl.logging import WandbLogger


def _fake_run():
    return SimpleNamespace(
        config=SimpleNamespace(update=lambda *args, **kwargs: None),
        define_metric=lambda name, step_metric: None,
        log=lambda values: None,
        finish=lambda: None,
    )


def _fake_wandb(run):
    return SimpleNamespace(
        init=lambda **kwargs: run,
        Html=lambda value: value,
        Image=lambda value: value,
    )


def test_wandb_logger_forwards_metrics_and_context(monkeypatch):
    logged = []
    run = _fake_run()
    run.log = lambda values: logged.append(values)
    fake_wandb = _fake_wandb(run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger(argparse.Namespace(gamma=0.99), "test-run")
    logger.track(3.0, name="loss", step=7, context={"subset": "train"})

    assert logged == [{"loss/subset=train": 3.0, "loss/subset=train_step": 7}]


def test_wandb_logger_uses_default_project(monkeypatch):
    init_kwargs = []
    run = _fake_run()
    fake_wandb = _fake_wandb(run)
    fake_wandb.init = lambda **kwargs: init_kwargs.append(kwargs) or run
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    WandbLogger(argparse.Namespace(gamma=0.99), "test-run")

    assert init_kwargs[0]["project"] == "restaurant_rl_factored"


def test_wandb_logger_loads_dotenv_when_available(monkeypatch):
    load_calls = []
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(load_dotenv=lambda: load_calls.append(True)),
    )
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(_fake_run()))

    WandbLogger(argparse.Namespace(gamma=0.99), "test-run")

    assert load_calls == [True]


def test_wandb_logger_works_without_dotenv(monkeypatch):
    # dotenv absent -> import fails -> training still initializes wandb.
    monkeypatch.delitem(sys.modules, "dotenv", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(_fake_run()))

    logger = WandbLogger(argparse.Namespace(gamma=0.99), "test-run")

    assert logger._run is not None


def test_wandb_logger_disabled_on_init_failure(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("wandb: API key not configured (WANDB_API_KEY)")

    fake_wandb = SimpleNamespace(
        init=boom,
        Html=lambda value: value,
        Image=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger(argparse.Namespace(gamma=0.99), "test-run")

    # No run -> all track/close methods must be safe no-ops.
    assert logger._run is None
    logger.track(1.0, name="loss", step=1)
    logger.track_text("note", name="note", step=1)
    logger.track_image(Path("/tmp/x.png"), name="img", step=1)
    logger.close()
