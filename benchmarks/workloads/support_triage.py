"""
Support-triage workload — three @ta.agent steps:

  ticket → classifier → retriever → drafter

- classifier: ticket text → one of CATEGORIES
- retriever: ticket text + same-category candidates → 3 most-similar IDs
- drafter: ticket text + retrieved candidates → a draft response

Designed so:

- Each agent is small enough to keep per-call cost low (~$0.001 at Haiku).
- The drafter's prompt depends on the retriever's output, so a downstream
  invalidation case exists (used by the research_analyst replay test;
  noted here for symmetry).
- Identical ticket text → identical inputs on the second pass → exercise
  ThriftAI's exact cache deterministically.
- Paraphrastic variants in the same cluster → exercise the semantic cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import thriftai as ta

CATEGORIES = [
    "billing",
    "technical",
    "shipping",
    "account",
    "feature_request",
    "feedback",
    "refund",
    "outage",
    "security",
    "other",
]


@ta.agent(name="classifier")
def classify(run, ticket_text: str, model: str) -> str:
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer-support triage system. Classify the "
                    "user's ticket into exactly one of these categories: "
                    + ", ".join(CATEGORIES)
                    + ". Reply with only the category name, lowercase, no other text."
                ),
            },
            {"role": "user", "content": ticket_text},
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="retriever", depends_on=["classifier"])
def retrieve(
    run, ticket_text: str, candidates: list[dict], model: str
) -> str:
    """Returns a comma-separated string of 3 ticket IDs."""
    formatted = "\n".join(
        f"[{c['id']}] {c['text']}" for c in candidates
    )
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Find the 3 past tickets most similar to the new ticket. "
                    "Reply with only the 3 ticket IDs, comma-separated, no other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"NEW TICKET:\n{ticket_text}\n\n"
                    f"CANDIDATE PAST TICKETS:\n{formatted}"
                ),
            },
        ],
        model=model,
        temperature=0.0,
    )


@ta.agent(name="drafter", depends_on=["retriever"])
def draft_response(
    run,
    ticket_text: str,
    retrieved_ids: str,
    corpus: list[dict],
    model: str,
) -> str:
    ids = {s.strip() for s in retrieved_ids.split(",") if s.strip()}
    examples = "\n\n".join(
        f"[{t['id']}] {t['text']}" for t in corpus if t["id"] in ids
    )
    return run.completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You draft helpful customer-support responses. Be concise, "
                    "directly address the issue, and reference how similar past "
                    "tickets were handled when relevant. 2-4 sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"NEW TICKET:\n{ticket_text}\n\n"
                    f"REFERENCE TICKETS:\n{examples}"
                ),
            },
        ],
        model=model,
        temperature=0.0,
    )


# ---- workload-level orchestration -----------------------------------------


def load_tasks(path: Path) -> list[dict]:
    """Load tickets from a JSONL file. Each line: {id, category, cluster, text}."""
    tasks: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def candidates_for(ticket: dict, corpus: list[dict], limit: int = 5) -> list[dict]:
    """Same-category candidates, excluding the ticket itself, capped at limit."""
    same = [c for c in corpus if c["category"] == ticket["category"] and c["id"] != ticket["id"]]
    return same[:limit]


def run_one(session, ticket: dict, corpus: list[dict], model: str) -> dict:
    """Run the three-agent pipeline for one ticket. Returns artifacts for the judge."""
    candidates = candidates_for(ticket, corpus)
    with session.run() as run:
        category = classify(run, ticket["text"], model)
        retrieved = retrieve(run, ticket["text"], candidates, model)
        draft = draft_response(run, ticket["text"], retrieved, corpus, model)
    return {
        "task_id": ticket["id"],
        "category": category,
        "retrieved": retrieved,
        "draft": draft,
    }
