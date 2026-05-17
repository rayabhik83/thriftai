"""
End-to-end smoke test against a real LLM provider.

The unit tests in this repo all mock LiteLLM. That catches our own bugs
but misses an entire class: provider response-shape drift, LiteLLM
version bumps, pricing-table desync, real auth failures. This test is
the canary — one cold pipeline run + one warm re-run against a live
model.

**Skipped by default.** Set `THRIFTAI_LIVE_TEST=1` and provide the
relevant API key (e.g. `ANTHROPIC_API_KEY`) to run. CI runs it nightly
via `.github/workflows/live.yml`.

Cost cap: with the default `claude-haiku-4-5` model and the trivial
prompts here, one full run (cold + warm) is well under $0.01.
"""

from __future__ import annotations

import os

import pytest

import thriftai as ta
from thriftai.agent import _agent_registry


LIVE = os.environ.get("THRIFTAI_LIVE_TEST") == "1"
MODEL = os.environ.get("THRIFTAI_LIVE_MODEL", "anthropic/claude-haiku-4-5")


@pytest.fixture(autouse=True)
def _reset_registry():
    _agent_registry.clear()
    yield
    _agent_registry.clear()


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE,
    reason="set THRIFTAI_LIVE_TEST=1 (and an LLM API key) to run",
)
def test_three_agent_pipeline_against_real_provider(tmp_path):
    """Cold run hits the real provider; warm re-run is fully cached."""

    @ta.agent(name="researcher")
    def research(s, topic: str) -> str:
        return s.completion(
            messages=[
                {"role": "system", "content": "Reply in one short sentence."},
                {"role": "user", "content": f"Define: {topic}"},
            ],
            model=MODEL,
        )

    @ta.agent(name="analyzer", depends_on=["researcher"])
    def analyze(s, raw: str) -> str:
        return s.completion(
            messages=[
                {"role": "system", "content": "Reply in one short sentence."},
                {"role": "user", "content": f"Is this accurate?: {raw}"},
            ],
            model=MODEL,
        )

    @ta.agent(name="writer", depends_on=["analyzer"])
    def write(s, analysis: str) -> str:
        return s.completion(
            messages=[
                {"role": "system", "content": "Reply in one short sentence."},
                {"role": "user", "content": f"Restate: {analysis}"},
            ],
            model=MODEL,
        )

    session = ta.Session(cache_dir=tmp_path)

    # --- Cold run: every agent should hit the live provider ---------------
    with session.run() as run:
        result = write(run, analyze(run, research(run, "memoization")))
        cold_trace_id = run.trace_id

    assert isinstance(result, str) and result.strip(), \
        "writer returned empty; provider may be returning a malformed shape"
    assert len(run.cost_report.entries) == 3
    assert [e.agent_name for e in run.cost_report.entries] == [
        "researcher", "analyzer", "writer",
    ]
    assert all(e.resolution == "live" for e in run.cost_report.entries), \
        f"expected all live, got {[e.resolution for e in run.cost_report.entries]}"
    assert run.cost_report.total_cost > 0, \
        "live run reported $0 — pricing table desync or tokens not surfaced"

    # Trace was actually persisted to disk.
    saved = session.trace_store.load(cold_trace_id)
    assert len(saved.entries) == 3
    assert all(e.response_text for e in saved.entries)

    # --- Warm run: identical inputs, every call should resolve from cache ---
    with session.run() as run_warm:
        result_warm = write(run_warm, analyze(run_warm, research(run_warm, "memoization")))

    # Cache returns the exact same text the live call produced.
    assert result_warm == result
    assert all(e.resolution == "cache_hit" for e in run_warm.cost_report.entries), \
        f"warm run hit provider unexpectedly: "\
        f"{[e.resolution for e in run_warm.cost_report.entries]}"
    assert run_warm.cost_report.total_cost == 0
    assert run_warm.cost_report.total_saved > 0
