"""
Synthesize the canonical research-analyst question corpus.

Output: 20 fixed research questions covering finance, science, history,
and current events. The category mix is deliberate — semantic-cache
wrong-hit risk is most visible when distinctly different questions
share surface vocabulary.

Output is committed at benchmarks/data/research_questions.jsonl and is
the canonical input for the workload. Re-run this script only to
intentionally regenerate.

Cost: ~$0.05 of Sonnet API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

BENCH_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BENCH_DIR / "data" / "research_questions.jsonl"


# 4 categories × 5 questions = 20. Categories are intentionally diverse
# so paraphrastic-but-different questions stress the semantic cache.
CATEGORIES: list[dict] = [
    {
        "name": "finance",
        "prompt": (
            "Generate 5 distinct medium-complexity research questions in finance. "
            "Mix corporate finance, monetary policy, market microstructure, and "
            "personal finance. Each question should require multi-step analysis "
            "(not lookup) and be specific enough to answer concretely."
        ),
    },
    {
        "name": "science",
        "prompt": (
            "Generate 5 distinct medium-complexity research questions in science. "
            "Mix biology, physics, chemistry, and earth science. Each question "
            "should require synthesis of multiple concepts and avoid yes/no answers."
        ),
    },
    {
        "name": "history",
        "prompt": (
            "Generate 5 distinct medium-complexity research questions in history. "
            "Mix political, economic, social, and intellectual history across "
            "different eras and regions. Avoid trivia questions; favor causal "
            "or comparative questions."
        ),
    },
    {
        "name": "current_events",
        "prompt": (
            "Generate 5 distinct medium-complexity research questions about "
            "ongoing topics as of 2026: AI policy, climate adaptation, the "
            "economics of geographic migration, public health systems, or "
            "energy transition. Avoid time-pegged trivia; favor analytical "
            "questions that an analyst could still address with reasoning."
        ),
    },
]


SYSTEM_PROMPT = """\
You generate realistic research questions for an analyst benchmark.
Output ONLY a JSON array of exactly 5 strings, each a single research
question (no question numbers, no markdown). Questions should be:

- One sentence each, ending in a question mark.
- Substantive enough to require multi-step analysis to answer well.
- Specific (named entities, named timeframes, concrete metrics) so two
  competent analysts would converge on similar approaches.

No surrounding text, no markdown fences — just the JSON array.
"""


def synthesize_category(litellm_module, cat: dict, model: str) -> list[str]:
    response = litellm_module.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cat["prompt"]},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    questions = json.loads(text)
    if not isinstance(questions, list) or len(questions) != 5:
        raise RuntimeError(
            f"Expected list of 5 strings for {cat['name']}, got: {questions!r}"
        )
    return [str(q).strip() for q in questions]


def main() -> int:
    parser = argparse.ArgumentParser(prog="research_questions_gen")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    load_dotenv(BENCH_DIR / ".env")
    import litellm  # noqa: F401

    questions: list[dict] = []
    next_id = 1
    for cat in CATEGORIES:
        print(f"  category: {cat['name']}", file=sys.stderr)
        items = synthesize_category(litellm, cat, args.model)
        for q in items:
            questions.append(
                {
                    "id": f"Q{next_id:03d}",
                    "category": cat["name"],
                    "question": q,
                }
            )
            next_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"wrote {args.out} ({len(questions)} questions across {len(CATEGORIES)} categories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
