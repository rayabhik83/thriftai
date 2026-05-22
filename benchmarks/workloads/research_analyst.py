"""
Research-analyst workload — four-agent pipeline:

  question → scout → planner → analyst → critic

- scout (depends_on []): generate 5 sub-questions / angles to investigate
- planner (depends_on [scout]): structure the research approach
- analyst (depends_on [planner]): produce the main analysis
- critic (depends_on [analyst]): critique the analysis and propose
  the strongest counter-argument

`depends_on` chain is intentional. With ThriftAI's downstream-
invalidation mechanism, changing one agent late in the chain (e.g.
the critic) lets you iterate without re-running scout/planner/analyst —
this is the dev-loop story the library was built for.

The brief asked for a fourth condition (thriftai_replay) on this
workload specifically. Adding that condition is wired in the runner;
this workload's `run_one` is parametric on the context-manager factory
so the runner can swap in `session.replay(trace_id, live=["critic"])`
for the measured pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import thriftai as ta


@ta.agent(name="scout")
def scout(run, question: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research scout. Given a research question, "
                    "list exactly 5 distinct sub-questions or angles worth "
                    "investigating to answer it well. One per line, no numbering."
                ),
            },
            {"role": "user", "content": question},
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="planner", depends_on=["scout"])
def planner(run, question: str, scout_output: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. Given a question and a list "
                    "of sub-questions, produce a 3-4 step plan for answering "
                    "the question concretely. Each step is one sentence. "
                    "No preamble, no numbered list — just the steps separated "
                    "by newlines."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nSub-questions:\n{scout_output}",
            },
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="analyst", depends_on=["planner"])
def analyst(run, question: str, plan: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful research analyst. Given a question and a "
                    "plan, write a 4-6 sentence analysis answering the question. "
                    "Cite specific mechanisms, numbers, or examples where they "
                    "exist. Avoid hedging language."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nPlan:\n{plan}",
            },
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="critic", depends_on=["analyst"])
def critic(run, question: str, analysis: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a tough but constructive critic. Given a question "
                    "and an analyst's answer, write a 2-3 sentence critique that "
                    "names the single strongest counter-argument or unaddressed "
                    "factor. Be specific."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nAnalysis:\n{analysis}",
            },
        ],
        model=model,
        temperature=0.0,
    )


# ---- workload-level orchestration -----------------------------------------


def load_tasks(path: Path) -> list[dict]:
    """Each line: {id, category, question}."""
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
    corpus: list[dict],  # noqa: ARG001 — unused, present for signature symmetry
    model: str,
    *,
    replay_trace_id: str | None = None,
    live_agents: list[str] | None = None,
) -> dict:
    """Run the four-agent pipeline for one question.

    Replay support: if `replay_trace_id` is passed, the call uses
    `session.replay(trace_id, live=live_agents)` instead of
    `session.run()`. The runner sets these for the `thriftai_replay`
    condition.
    """
    if replay_trace_id is not None:
        ctx = session.replay(trace_id=replay_trace_id, live=live_agents or [])
    else:
        ctx = session.run()

    with ctx as run:
        s = scout(run, task["question"], model)
        p = planner(run, task["question"], s, model)
        a = analyst(run, task["question"], p, model)
        c = critic(run, task["question"], a, model)
        trace_id = run.trace_id

    return {
        "task_id": task["id"],
        "scout": s,
        "plan": p,
        "analysis": a,
        "critique": c,
        "trace_id": trace_id,
    }
