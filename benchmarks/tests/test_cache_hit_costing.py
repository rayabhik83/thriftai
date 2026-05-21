"""
Verify the headline table treats cache_hit / semantic_hit / replay as $0
paid and surfaces the would-have cost as savings.
"""

from __future__ import annotations

from benchmarks.runner import report


_PRICING = {
    "pulled_on": "2026-05-19",
    "models": {
        "claude-haiku-4-5": {
            "input_per_million_usd": 0.25,
            "output_per_million_usd": 1.25,
        },
    },
}


def _record(
    *,
    resolution: str,
    task_id: str,
    agent_name: str,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    run_id: str = "rid",
    seed: int = 0,
) -> dict:
    return {
        "timestamp": "2026-05-20T00:00:00Z",
        "run_id": run_id,
        "workload": "support_triage",
        "condition": "thriftai_warm",
        "model_under_test": "claude-haiku-4-5",
        "task_id": task_id,
        "seed": seed,
        "agent_name": agent_name,
        "model": "claude-haiku-4-5",
        "broker_resolution": resolution,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_cost_usd_litellm": 0.0,
        "would_have_cost_usd_litellm": 0.0005,
        "embedding_cost_usd_litellm": 0.0,
        "similarity_score": None,
        "latency_total_ms": 1.0,
        "latency_api_ms": None,
        "latency_overhead_ms": 1.0,
        "response_text_hash": "sha256:abc",
    }


def test_cache_hits_show_zero_paid_and_nonzero_saved():
    """All-cache-hit cell: paid is $0, saved equals would-have-cost."""
    records = [
        _record(resolution="cache_hit", task_id="T001", agent_name="classifier"),
        _record(resolution="cache_hit", task_id="T001", agent_name="retriever"),
        _record(resolution="cache_hit", task_id="T001", agent_name="drafter"),
    ]
    paid, would_have = report._per_task_cost(records, _PRICING)
    cell = ("support_triage", "thriftai_warm", "claude-haiku-4-5")
    assert paid[cell] == [0.0]
    # 3 calls × (1000/M × $0.25 + 200/M × $1.25) = 3 × $0.0005 = $0.0015
    assert would_have[cell] == [pytest_approx(0.0015)]


def test_mixed_live_and_cache_in_same_task():
    """One live + two cache_hit: paid = one call's cost; saved = two calls' cost."""
    records = [
        _record(resolution="live", task_id="T001", agent_name="classifier"),
        _record(resolution="cache_hit", task_id="T001", agent_name="retriever"),
        _record(resolution="cache_hit", task_id="T001", agent_name="drafter"),
    ]
    paid, would_have = report._per_task_cost(records, _PRICING)
    cell = ("support_triage", "thriftai_warm", "claude-haiku-4-5")
    # Paid = 1 live call = $0.0005
    assert paid[cell] == [pytest_approx(0.0005)]
    # Would-have = all 3 = $0.0015
    assert would_have[cell] == [pytest_approx(0.0015)]


def test_render_includes_saved_column():
    records = [
        _record(resolution="cache_hit", task_id=f"T{i}", agent_name="classifier")
        for i in range(3)
    ]
    out = report.render(records, _PRICING)
    assert "saved" in out.lower()
    # All-zero paid for 3 cache hits.
    assert "0.0000 ± 0.0000" in out


# Local approx helper to avoid pulling pytest into a non-pytest namespace.
def pytest_approx(value, rel=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= rel * max(abs(value), 1.0)
        def __repr__(self):
            return f"~{value}"
    return _Approx()
