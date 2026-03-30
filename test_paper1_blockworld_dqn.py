from __future__ import annotations

from pathlib import Path

from anticipatory_rl.agents.paper1_blockworld_image_dqn import build_parser, train
from anticipatory_rl.agents.paper1_blockworld_image_dqn_infer import run_compare


def _make_train_args(tasks_per_reset: int):
    return build_parser().parse_args(
        [
            "--total-steps",
            "12",
            "--replay-size",
            "32",
            "--batch-size",
            "4",
            "--hidden-dim",
            "64",
            "--num-envs",
            "1",
            "--task-library-size",
            "8",
            "--max-task-steps",
            "3",
            "--render-tile-px",
            "8",
            "--tasks-per-reset",
            str(tasks_per_reset),
            "--env-reset-tasks",
            "10",
            "--seed",
            "0",
        ]
    )


def test_train_smoke_myopic_and_anticipatory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    myo_ckpt = train(_make_train_args(tasks_per_reset=1))
    ant_ckpt = train(_make_train_args(tasks_per_reset=10))
    assert myo_ckpt.exists()
    assert ant_ckpt.exists()
    assert (myo_ckpt.parent / "train_summary.json").exists()
    assert (ant_ckpt.parent / "train_summary.json").exists()
    assert (myo_ckpt.parent / "task_records.json").exists()
    assert (ant_ckpt.parent / "task_records.json").exists()


def test_compare_infer_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ant_ckpt = train(_make_train_args(tasks_per_reset=10))
    myo_ckpt = train(_make_train_args(tasks_per_reset=1))

    class Args:
        anticipatory_weights = ant_ckpt
        myopic_weights = myo_ckpt
        output_dir = Path("runs") / "compare_blockworld_image_dqn_infer"
        num_sequences = 2
        tasks_per_reset = 10
        total_steps = 200
        hidden_dim = 64
        gamma = 0.99
        seed = 0
        softmax_temperature = 0.0
        task_library_size = 8
        max_task_steps = 3
        success_reward = 12.0
        step_penalty = 1.0
        invalid_action_penalty = 5.0
        correct_pick_bonus = 1.0
        render_tile_px = 8
        render_margin_px = None
        procedural_layout = True

    run_compare(Args())
    comparison_json = Args.output_dir / "comparison.json"
    assert comparison_json.exists()
    contents = comparison_json.read_text(encoding="utf-8")
    assert "avg_total_sequence_cost" in contents
    assert "cost_by_task_index" in contents
