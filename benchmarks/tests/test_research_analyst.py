"""Verify the research_analyst pipeline against a mocked litellm."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.runner import instrumentation
from benchmarks.workloads import research_analyst


def _fake_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 30):
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


def test_run_one_invokes_four_agents(tmp_path: Path):
    """scout → planner → analyst → critic, in that order."""
    from thriftai import Session

    session = Session(cache_dir=tmp_path / "ta_cache", enabled=False)

    responses = iter([
        _fake_response("angle 1\nangle 2\nangle 3"),
        _fake_response("step 1\nstep 2\nstep 3"),
        _fake_response("Analysis: yes."),
        _fake_response("Critique: counter-point."),
    ])

    with patch("litellm.completion", side_effect=lambda *a, **k: next(responses)), patch(
        "litellm.completion_cost", return_value=0.0001
    ):
        result = research_analyst.run_one(
            session,
            {"id": "Q001", "category": "test", "question": "Why?"},
            corpus=[],
            model="claude-haiku-4-5",
        )

    assert result["task_id"] == "Q001"
    assert result["scout"].startswith("angle")
    assert result["plan"].startswith("step")
    assert "Analysis" in result["analysis"]
    assert "counter-point" in result["critique"]
    # trace_id is populated even with enabled=False (RunContext sets it
    # regardless; just no trace is recorded to disk).
    assert isinstance(result["trace_id"], str) and len(result["trace_id"]) > 0


def test_run_one_records_four_jsonl_lines_under_bench_context(tmp_path: Path):
    from thriftai import Session

    instrumentation.configure_output(tmp_path / "calls.jsonl")
    instrumentation.set_context(
        instrumentation.BenchContext(
            run_id="rid_research",
            workload="research_analyst",
            condition="baseline",
            model_under_test="claude-haiku-4-5",
            task_id="Q001",
            seed=0,
        )
    )

    session = Session(cache_dir=tmp_path / "ta_cache_b", enabled=False)
    responses = iter([_fake_response(f"agent {i} output") for i in range(4)])

    with patch("litellm.completion", side_effect=lambda *a, **k: next(responses)), patch(
        "litellm.completion_cost", return_value=0.0001
    ):
        research_analyst.run_one(
            session,
            {"id": "Q001", "category": "test", "question": "Why?"},
            corpus=[],
            model="claude-haiku-4-5",
        )

    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    import json
    agents = [json.loads(line)["agent_name"] for line in lines]
    assert agents == ["scout", "planner", "analyst", "critic"]
