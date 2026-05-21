"""
Smoke-test the plot module — generates the three figures from a tiny
synthetic record set and checks the PNG files end up on disk.

We don't validate pixel content (matplotlib visual regression is fragile);
we just confirm the functions run cleanly and produce non-empty files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.runner import plots, report


_PRICING = {
    "models": {
        "claude-haiku-4-5": {
            "input_per_million_usd": 0.25,
            "output_per_million_usd": 1.25,
        },
    },
}


def _synthetic_records():
    """Two cells: baseline (all live, $$) and thriftai_warm (all cache_hit, $0)."""
    records = []
    for seed in range(2):
        for ticket in range(5):
            # Baseline: 3 live calls per ticket.
            for agent in ("classifier", "retriever", "drafter"):
                records.append({
                    "timestamp": "2026-05-20T00:00:00Z",
                    "run_id": f"rid_baseline_s{seed}",
                    "workload": "support_triage",
                    "condition": "baseline",
                    "model_under_test": "claude-haiku-4-5",
                    "task_id": f"T{ticket:03d}",
                    "seed": seed,
                    "agent_name": agent,
                    "model": "claude-haiku-4-5",
                    "broker_resolution": "live",
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "actual_cost_usd_litellm": 0.000175,
                    "would_have_cost_usd_litellm": 0.000175,
                    "embedding_cost_usd_litellm": 0.0,
                    "similarity_score": None,
                    "latency_total_ms": 800 + ticket * 10,
                    "latency_api_ms": 750,
                    "latency_overhead_ms": 50,
                    "response_text_hash": "sha256:x",
                })
            # Warm: 3 cache_hit calls per ticket.
            for agent in ("classifier", "retriever", "drafter"):
                records.append({
                    "timestamp": "2026-05-20T00:00:01Z",
                    "run_id": f"rid_warm_s{seed}",
                    "workload": "support_triage",
                    "condition": "thriftai_warm",
                    "model_under_test": "claude-haiku-4-5",
                    "task_id": f"T{ticket:03d}",
                    "seed": seed,
                    "agent_name": agent,
                    "model": "claude-haiku-4-5",
                    "broker_resolution": "cache_hit",
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "actual_cost_usd_litellm": 0.0,
                    "would_have_cost_usd_litellm": 0.000175,
                    "embedding_cost_usd_litellm": 0.0,
                    "similarity_score": None,
                    "latency_total_ms": 1,
                    "latency_api_ms": None,
                    "latency_overhead_ms": 1,
                    "response_text_hash": "sha256:y",
                })
    return records


def test_cost_reduction_writes_png(tmp_path: Path):
    by_cell = plots._per_cell_aggregates(_synthetic_records(), _PRICING)
    out = plots.plot_cost_reduction(by_cell, tmp_path / "cost.png")
    assert out.exists() and out.stat().st_size > 0


def test_latency_writes_png(tmp_path: Path):
    by_cell = plots._per_cell_aggregates(_synthetic_records(), _PRICING)
    out = plots.plot_latency_distribution(by_cell, tmp_path / "lat.png")
    assert out.exists() and out.stat().st_size > 0


def test_pareto_writes_png_without_judge_data(tmp_path: Path):
    """Pareto requires judge data; with none it should produce an empty-ish file but not crash."""
    by_cell = plots._per_cell_aggregates(_synthetic_records(), _PRICING)
    out = plots.plot_quality_cost_pareto(by_cell, {}, tmp_path / "pareto.png")
    assert out.exists() and out.stat().st_size > 0


def test_pareto_with_judge_data(tmp_path: Path):
    by_cell = plots._per_cell_aggregates(_synthetic_records(), _PRICING)
    judge = {
        ("support_triage", "baseline", "claude-haiku-4-5"): [4.0, 4.5, 5.0],
        ("support_triage", "thriftai_warm", "claude-haiku-4-5"): [4.0, 4.5, 5.0],
    }
    out = plots.plot_quality_cost_pareto(by_cell, judge, tmp_path / "pareto.png")
    assert out.exists() and out.stat().st_size > 0


def test_main_with_no_data_is_clean(tmp_path: Path, monkeypatch):
    """`python -m benchmarks.runner.plots` on an empty repo just no-ops."""
    monkeypatch.setattr(plots, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(plots, "PLOTS_DIR", tmp_path / "plots")
    # report.RAW_DIR is what load_calls actually consults — patch it too.
    monkeypatch.setattr(report, "RAW_DIR", tmp_path / "raw")
    rc = plots.main()
    assert rc == 0
