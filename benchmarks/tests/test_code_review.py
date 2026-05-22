"""Verify the code_review pipeline against a mocked litellm."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.runner import instrumentation
from benchmarks.workloads import code_review


def _fake_response(content: str, prompt_tokens: int = 200, completion_tokens: int = 80):
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


def test_run_one_invokes_three_agents(tmp_path: Path):
    from thriftai import Session

    session = Session(cache_dir=tmp_path / "ta_cache", enabled=False)
    responses = iter([
        _fake_response("L3-7: missing input validation"),
        _fake_response("@@ -3,5 +3,7 @@\n+if x is None: return\n"),
        _fake_response("Critique: doesn't handle empty list case."),
    ])

    with patch("litellm.completion", side_effect=lambda *a, **k: next(responses)), patch(
        "litellm.completion_cost", return_value=0.0002
    ):
        result = code_review.run_one(
            session,
            {"id": "C001", "code": "def f(x): return x[0]"},
            corpus=[],
            model="claude-haiku-4-5",
        )

    assert result["task_id"] == "C001"
    assert "validation" in result["issues"]
    assert result["patch"].startswith("@@")
    assert "empty list" in result["critique"]


def test_run_one_records_three_jsonl_lines(tmp_path: Path):
    from thriftai import Session

    instrumentation.configure_output(tmp_path / "calls.jsonl")
    instrumentation.set_context(
        instrumentation.BenchContext(
            run_id="rid_cr",
            workload="code_review",
            condition="baseline",
            model_under_test="claude-haiku-4-5",
            task_id="C001",
            seed=0,
        )
    )

    session = Session(cache_dir=tmp_path / "ta_cache_b", enabled=False)
    responses = iter([_fake_response(f"out {i}") for i in range(3)])

    with patch("litellm.completion", side_effect=lambda *a, **k: next(responses)), patch(
        "litellm.completion_cost", return_value=0.0001
    ):
        code_review.run_one(
            session,
            {"id": "C001", "code": "def f(): pass"},
            corpus=[],
            model="claude-haiku-4-5",
        )

    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    agents = [json.loads(line)["agent_name"] for line in lines]
    assert agents == ["reviewer", "proposer", "self_critic"]
