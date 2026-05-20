"""
Verify the support_triage workload structure end-to-end with a mocked
litellm. No API calls; runs in CI without a key.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.runner import instrumentation
from benchmarks.workloads import support_triage

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "support_tickets_tiny.jsonl"


def _fake_response(content: str, prompt_tokens: int = 30, completion_tokens: int = 5):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return response


@pytest.fixture(autouse=True)
def installed_patches():
    instrumentation.install()
    yield
    instrumentation.uninstall()
    instrumentation.set_context(None)


def test_load_tasks_reads_fixture():
    tasks = support_triage.load_tasks(FIXTURE_PATH)
    assert len(tasks) == 5
    assert all({"id", "category", "cluster", "text"} <= set(t) for t in tasks)


def test_candidates_for_filters_by_category_and_excludes_self():
    corpus = support_triage.load_tasks(FIXTURE_PATH)
    ticket = corpus[0]  # T001 billing
    candidates = support_triage.candidates_for(ticket, corpus, limit=5)
    assert all(c["category"] == "billing" for c in candidates)
    assert all(c["id"] != "T001" for c in candidates)
    # Fixture has 3 billing tickets total → 2 candidates after self-exclusion.
    assert len(candidates) == 2


def test_run_one_invokes_three_agents_in_order(tmp_path: Path):
    """The pipeline calls classifier, retriever, drafter in sequence."""
    from thriftai import Session

    corpus = support_triage.load_tasks(FIXTURE_PATH)
    session = Session(cache_dir=tmp_path / "thriftai_cache", enabled=False)

    # Sequence of mock responses for the three agent calls.
    responses = iter(
        [
            _fake_response("billing"),               # classifier
            _fake_response("T002, T003, T001"),     # retriever
            _fake_response("Sorry about the duplicate charge..."),  # drafter
        ]
    )

    def _next_response(*args, **kwargs):
        return next(responses)

    with patch("litellm.completion", side_effect=_next_response), patch(
        "litellm.completion_cost", return_value=0.0001
    ):
        result = support_triage.run_one(session, corpus[0], corpus, model="claude-haiku-4-5")

    assert result["task_id"] == "T001"
    assert result["category"] == "billing"
    assert result["retrieved"].startswith("T002")
    assert "duplicate charge" in result["draft"]


def test_run_one_with_bench_context_writes_jsonl(tmp_path: Path):
    """When a bench context is active, three JSONL lines get written."""
    from thriftai import Session

    instrumentation.configure_output(tmp_path / "calls.jsonl")
    instrumentation.set_context(
        instrumentation.BenchContext(
            run_id="rid_test",
            workload="support_triage",
            condition="baseline",
            model_under_test="claude-haiku-4-5",
            task_id="T001",
            seed=0,
        )
    )

    corpus = support_triage.load_tasks(FIXTURE_PATH)
    session = Session(cache_dir=tmp_path / "thriftai_cache", enabled=False)

    responses = iter(
        [
            _fake_response("billing"),
            _fake_response("T002, T003"),
            _fake_response("draft reply"),
        ]
    )

    with patch("litellm.completion", side_effect=lambda *a, **kw: next(responses)), patch(
        "litellm.completion_cost", return_value=0.0001
    ):
        support_triage.run_one(session, corpus[0], corpus, model="claude-haiku-4-5")

    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    import json
    records = [json.loads(line) for line in lines]
    assert [r["agent_name"] for r in records] == ["classifier", "retriever", "drafter"]
    assert all(r["task_id"] == "T001" for r in records)
    assert all(r["condition"] == "baseline" for r in records)
