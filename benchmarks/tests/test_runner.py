"""Verify the runner's argument parsing and missing-data handling."""

from __future__ import annotations

from pathlib import Path


from benchmarks.runner import run as runner_module


def test_parse_args_minimal():
    args = runner_module.parse_args(["--workload", "support_triage"])
    assert args.workload == "support_triage"
    assert args.n == 5
    assert args.model is None


def test_parse_args_with_overrides():
    args = runner_module.parse_args([
        "--workload", "support_triage",
        "--n", "2",
        "--model", "claude-haiku-4-5",
        "--task-limit", "3",
    ])
    assert args.n == 2
    assert args.model == "claude-haiku-4-5"
    assert args.task_limit == 3


def test_main_skips_missing_data_file(tmp_path: Path, monkeypatch, capsys):
    """A workload whose data_file doesn't exist must skip cleanly, not crash."""
    # Point the runner at a temp tree with config but no data file.
    monkeypatch.setattr(runner_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "BENCH_DIR", tmp_path / "benchmarks")
    monkeypatch.setattr(runner_module, "RESULTS_DIR", tmp_path / "benchmarks" / "results")
    monkeypatch.setattr(runner_module, "RAW_DIR", tmp_path / "benchmarks" / "results" / "raw")
    monkeypatch.setattr(runner_module, "CACHE_ROOT", tmp_path / "benchmarks" / "cache" / "sessions")

    (tmp_path / "benchmarks" / "configs").mkdir(parents=True)
    (tmp_path / "benchmarks" / "configs" / "support_triage.yaml").write_text(
        "workload: support_triage\n"
        "data_file: benchmarks/data/missing.jsonl\n"
        "default_model: claude-haiku-4-5\n"
        "conditions: [baseline]\n"
    )

    # Patch report.main so we don't try to write a report from nothing.
    monkeypatch.setattr(runner_module, "render_report", lambda: None)

    rc = runner_module.main(["--workload", "support_triage", "--n", "1"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "SKIP" in err
    assert "missing.jsonl" in err
