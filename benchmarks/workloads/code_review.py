"""
Code-review workload — three @ta.agent steps:

  code → reviewer → proposer → self_critic

- reviewer: read the code and identify 3-5 concrete issues
- proposer: given the code and issues, suggest specific fixes
- self_critic: given the code + issues + fixes, identify what's wrong
  with the fixes (regressions, missed edge cases, style problems)

This is the closest thing to a real PR-review loop you can simulate
with a single LLM. Same caching dynamics as support_triage: identical
inputs → identical outputs → exact cache hits dominate the warm
condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import thriftai as ta


@ta.agent(name="reviewer")
def reviewer(run, code: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior code reviewer. Read the code and list "
                    "3-5 concrete, actionable issues. Each issue is one "
                    "sentence, prefixed with the line range it touches. "
                    "No preamble; just the bulleted list."
                ),
            },
            {"role": "user", "content": code},
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="proposer", depends_on=["reviewer"])
def proposer(run, code: str, issues: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior engineer proposing fixes. Given a piece "
                    "of code and a list of issues, write a single short patch "
                    "in unified-diff style addressing the top 2-3 issues. "
                    "No commentary outside the diff body."
                ),
            },
            {
                "role": "user",
                "content": f"CODE:\n{code}\n\nISSUES:\n{issues}",
            },
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="self_critic", depends_on=["proposer"])
def self_critic(run, code: str, issues: str, patch: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are critiquing your own proposed patch. Given the "
                    "original code, the identified issues, and the patch, "
                    "name the strongest regression-risk or unaddressed concern. "
                    "2-3 sentences, specific."
                ),
            },
            {
                "role": "user",
                "content": f"CODE:\n{code}\n\nISSUES:\n{issues}\n\nPATCH:\n{patch}",
            },
        ],
        model=model,
        temperature=0.0,
    )


# ---- workload-level orchestration -----------------------------------------


def load_tasks(path: Path) -> list[dict]:
    """Each line: {id, language, path, license, code}."""
    tasks: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run_one(
    session,
    task: dict,
    corpus: list[dict],  # noqa: ARG001
    model: str,
    *,
    replay_trace_id: str | None = None,
    live_agents: list[str] | None = None,
) -> dict:
    if replay_trace_id is not None:
        ctx = session.replay(trace_id=replay_trace_id, live=live_agents or [])
    else:
        ctx = session.run()

    with ctx as run:
        issues = reviewer(run, task["code"], model)
        patch = proposer(run, task["code"], issues, model)
        critique = self_critic(run, task["code"], issues, patch, model)
        trace_id = run.trace_id

    return {
        "task_id": task["id"],
        "issues": issues,
        "patch": patch,
        "critique": critique,
        "trace_id": trace_id,
    }
