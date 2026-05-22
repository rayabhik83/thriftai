"""
Verify the judge module's caching, rubric routing, and parse behavior
against a mocked litellm.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks.judge import llm_judge


def _fake_completion_returning(score_dict: dict, prompt_tokens: int = 200, completion_tokens: int = 50):
    """Build a fake `litellm.completion(...)` that returns score_dict as JSON."""
    import json as _json

    def fake_completion(*args, **kwargs):
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content=_json.dumps(score_dict)))
        ]
        response.usage = MagicMock(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return response

    return fake_completion


@pytest.fixture(autouse=True)
def isolated_judge_cache(tmp_path: Path, monkeypatch):
    """Point the judge cache at a tmp path so tests don't share state."""
    monkeypatch.setattr(llm_judge, "CACHE_DB", tmp_path / "judge.db")


def test_judge_calls_litellm_and_parses_json():
    score = {
        "classification_correct": 5,
        "retrieval_relevance": 4,
        "draft_helpful": 5,
        "rationale": "Strong all around.",
    }
    fake = _fake_completion_returning(score)
    out = llm_judge.judge(
        "support_triage",
        "T001",
        {"category": "billing", "retrieved": "T002,T003,T004", "draft": "Hi"},
        litellm_completion=fake,
    )
    assert out == score


def test_judge_caches_and_skips_second_call():
    """A second call with identical artifacts should not invoke litellm."""
    score = {"classification_correct": 4, "retrieval_relevance": 4, "draft_helpful": 4, "rationale": "ok"}
    calls = []

    def counting_fake(*args, **kwargs):
        calls.append(kwargs)
        return _fake_completion_returning(score)(*args, **kwargs)

    artifacts = {"category": "billing", "retrieved": "T2,T3,T4", "draft": "..."}

    out1 = llm_judge.judge("support_triage", "T001", artifacts, litellm_completion=counting_fake)
    out2 = llm_judge.judge("support_triage", "T001", artifacts, litellm_completion=counting_fake)

    assert out1 == out2 == score
    assert len(calls) == 1  # only the first call hit the fake


def test_judge_different_artifacts_no_cache_collision():
    score_a = {"classification_correct": 5, "retrieval_relevance": 5, "draft_helpful": 5, "rationale": "perfect"}
    score_b = {"classification_correct": 2, "retrieval_relevance": 2, "draft_helpful": 2, "rationale": "weak"}

    responses = iter([score_a, score_b])

    def routing_fake(*args, **kwargs):
        return _fake_completion_returning(next(responses))(*args, **kwargs)

    a = llm_judge.judge(
        "support_triage", "T001",
        {"category": "billing", "retrieved": "T2,T3,T4", "draft": "..."},
        litellm_completion=routing_fake,
    )
    b = llm_judge.judge(
        "support_triage", "T001",
        {"category": "shipping", "retrieved": "T5,T6,T7", "draft": "..."},
        litellm_completion=routing_fake,
    )
    assert a == score_a
    assert b == score_b


def test_judge_strips_markdown_fences():
    """Some models wrap JSON in ```json fences. Parser should strip them."""
    fake = MagicMock(side_effect=lambda *a, **k: _build_response_with_fences())
    out = llm_judge.judge(
        "support_triage",
        "T001",
        {"category": "x", "retrieved": "y", "draft": "z"},
        litellm_completion=fake,
    )
    assert out["classification_correct"] == 3


def _build_response_with_fences():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    "```json\n"
                    '{"classification_correct": 3, "retrieval_relevance": 3, "draft_helpful": 3, "rationale": "ok"}\n'
                    "```"
                )
            )
        )
    ]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=30)
    return response


def test_judge_raises_on_unknown_workload():
    with pytest.raises(ValueError, match="No rubric defined"):
        llm_judge.judge(
            "unknown_workload",
            "T001",
            {"foo": "bar"},
            litellm_completion=MagicMock(),
        )


def test_judge_artifacts_file_processes_jsonl(tmp_path: Path):
    """End-to-end: a JSONL of artifacts → list of per-task scores."""
    artifacts_jsonl = tmp_path / "artifacts.jsonl"
    import json
    with artifacts_jsonl.open("w") as f:
        for i in range(3):
            f.write(json.dumps({
                "task_id": f"T00{i+1}",
                "category": "billing",
                "retrieved": "T100,T101,T102",
                "draft": f"draft {i}",
            }) + "\n")

    scores_iter = iter([
        {"classification_correct": 5, "retrieval_relevance": 5, "draft_helpful": 5, "rationale": "a"},
        {"classification_correct": 3, "retrieval_relevance": 3, "draft_helpful": 3, "rationale": "b"},
        {"classification_correct": 1, "retrieval_relevance": 1, "draft_helpful": 1, "rationale": "c"},
    ])

    def fake_completion(*args, **kwargs):
        return _fake_completion_returning(next(scores_iter))(*args, **kwargs)

    # Manually patch the module-level lazy import path.
    import benchmarks.judge.llm_judge as judge_mod
    original_judge = judge_mod.judge

    def patched_judge(workload, task_id, artifacts, **kw):
        return original_judge(workload, task_id, artifacts, litellm_completion=fake_completion)

    import unittest.mock as mock
    with mock.patch.object(judge_mod, "judge", side_effect=patched_judge):
        results = llm_judge.judge_artifacts_file(artifacts_jsonl, "support_triage")

    assert len(results) == 3
    assert results[0]["task_id"] == "T001"
    assert results[2]["classification_correct"] == 1
