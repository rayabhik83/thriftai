"""
HumanEval slice — public benchmark wrapper.

Single-agent workload:
  prompt + function signature → completion → pass@1 (deterministic)

Unlike the other workloads we have no LLM-as-judge here; the quality
metric is whether the generated code passes the unit tests in the
official `human_eval` package. This proves ThriftAI doesn't perturb
correctness — baseline / cold / warm must yield identical pass@1 to
within noise (and at temperature=0 with deterministic models,
identical exactly).

Loading: `human_eval.data.read_problems()` exposes the full 164-task
set. We slice to the first 20 (stable order by task_id sort).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import thriftai as ta

SLICE_SIZE = 20


@ta.agent(name="completer")
def completer(run, prompt: str, model: str) -> str:
    """Generate the function body for a HumanEval prompt."""
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You complete Python functions. Given the docstring + "
                    "function signature, return ONLY the function body "
                    "(starting at the first line of the body, NOT including "
                    "the def line). No markdown fences, no comments before "
                    "code, no surrounding text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.0,
    )


# ---- workload-level orchestration -----------------------------------------


def load_tasks(path: Path) -> list[dict]:
    """Load the HumanEval slice from disk if we wrote one earlier; otherwise
    pull from the upstream package and write a snapshot.

    `path` is the canonical input file. If it doesn't exist, we create it
    once from `human_eval.data.read_problems()` so re-runs are deterministic
    against a frozen slice. Run `python -m benchmarks.workloads.humaneval`
    to force-regenerate.
    """
    path = Path(path)
    if not path.exists():
        problems = _read_upstream_problems()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_upstream_problems() -> list[dict]:
    from human_eval.data import read_problems

    raw = read_problems()
    # Stable order by task_id sort, take SLICE_SIZE.
    sorted_keys = sorted(raw.keys(), key=lambda k: (len(k), k))[:SLICE_SIZE]
    out: list[dict] = []
    for task_id in sorted_keys:
        p = raw[task_id]
        out.append(
            {
                "id": task_id.replace("/", "_"),
                "task_id": task_id,
                "prompt": p["prompt"],
                "entry_point": p["entry_point"],
                "test": p["test"],
                "canonical_solution": p.get("canonical_solution", ""),
            }
        )
    return out


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
        body = completer(run, task["prompt"], model)
        trace_id = run.trace_id

    # Reassemble: prompt already includes the def line; we just need the
    # body. Strip optional code fences.
    body_clean = _strip_fences(body)
    completion = task["prompt"] + body_clean

    return {
        "task_id": task["id"],
        "completion": completion,
        "raw_body": body,
        "trace_id": trace_id,
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (and optional language).
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Drop closing fence.
        last_fence = text.rfind("```")
        if last_fence != -1:
            text = text[:last_fence]
    return text.rstrip() + "\n"
