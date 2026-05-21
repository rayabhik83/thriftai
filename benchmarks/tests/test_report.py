"""
Verify the report renders cleanly with no data and with a tiny synthetic
record set. The empty-data path is critical — `make report` must not crash
before the first workload run.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.runner import report


_PRICING = {
    "pulled_on": "2026-05-19",
    "source_url": "https://example.invalid/pricing",
    "models": {
        "claude-haiku-4-5": {
            "input_per_million_usd": 0.25,
            "output_per_million_usd": 1.25,
        },
    },
}


def _write_raw(tmp_path: Path, run_id: str, records: list[dict]) -> Path:
    """Write a calls.jsonl under a fake run directory; return the directory."""
    run_dir = tmp_path / "raw" / run_id
    run_dir.mkdir(parents=True)
    jsonl = run_dir / "calls.jsonl"
    with jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return run_dir


def test_render_empty_has_section_headers():
    out = report.render([], _PRICING)
    assert "# ThriftAI Benchmark Results" in out
    assert "## Headline" in out
    assert "## Methodology" in out
    assert "_no data yet_" in out


def test_load_calls_handles_missing_directory(tmp_path: Path):
    nonexistent = tmp_path / "definitely-not-here"
    assert report.load_calls(nonexistent) == []


def test_render_with_records_includes_aggregates(tmp_path: Path):
    records = [
        {
            "timestamp": "2026-05-19T22:00:00Z",
            "run_id": "rid_A",
            "workload": "support_triage",
            "condition": "baseline",
            "model_under_test": "claude-haiku-4-5",
            "task_id": "ticket_0",
            "seed": 1,
            "agent_name": "classifier",
            "model": "claude-haiku-4-5",
            "broker_resolution": "live",
            "input_tokens": 1000,
            "output_tokens": 200,
            "actual_cost_usd_litellm": 0.0005,
            "would_have_cost_usd_litellm": 0.0005,
            "embedding_cost_usd_litellm": 0.0,
            "similarity_score": None,
            "latency_total_ms": 1500.0,
            "latency_api_ms": 1400.0,
            "latency_overhead_ms": 100.0,
            "response_text_hash": "sha256:abc",
        },
        # Second call in the same task — costs accumulate per (run_id, task_id)
        {
            "timestamp": "2026-05-19T22:00:01Z",
            "run_id": "rid_A",
            "workload": "support_triage",
            "condition": "baseline",
            "model_under_test": "claude-haiku-4-5",
            "task_id": "ticket_0",
            "seed": 1,
            "agent_name": "drafter",
            "model": "claude-haiku-4-5",
            "broker_resolution": "live",
            "input_tokens": 500,
            "output_tokens": 100,
            "actual_cost_usd_litellm": 0.0003,
            "would_have_cost_usd_litellm": 0.0003,
            "embedding_cost_usd_litellm": 0.0,
            "similarity_score": None,
            "latency_total_ms": 800.0,
            "latency_api_ms": 750.0,
            "latency_overhead_ms": 50.0,
            "response_text_hash": "sha256:def",
        },
    ]
    out = report.render(records, _PRICING)
    # Workload should appear in the headline table.
    assert "support_triage" in out
    assert "baseline" in out
    assert "claude-haiku-4-5" in out
    # Latency p50 from two calls (1500, 800) → 1150.
    assert "1150" in out
    # Resolution breakdown shows 2 live calls.
    assert "| support_triage | baseline | claude-haiku-4-5 | 2 |" in out
    # Both live → paid > 0, saved = 0.0000. Exact value across two calls:
    # 1000*0.25/M + 200*1.25/M + 500*0.25/M + 100*1.25/M = $0.00075.
    # Cost is per-task: both calls are the same task → one $0.00075 entry.
    # Headline column uses precision=4. The "saved" cell is 0.0000.
    assert "0.0008" in out  # paid 0.00075 → "0.0008" at precision=4
    assert "0.0000 ± 0.0000" in out  # saved column (all live)


def test_main_writes_file_when_no_data(tmp_path: Path, monkeypatch):
    """`python -m benchmarks.runner.report` should not crash on a fresh repo."""
    # Redirect both the raw dir and the report output to tmp_path.
    monkeypatch.setattr(report, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(report, "REPORT_PATH", tmp_path / "REPORT.md")
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text("pulled_on: 2026-05-19\nmodels: {}\n")
    monkeypatch.setattr(report, "PRICING_PATH", pricing_path)
    report.main()
    assert (tmp_path / "REPORT.md").exists()
    content = (tmp_path / "REPORT.md").read_text()
    assert "ThriftAI Benchmark Results" in content
