"""
Synthesize the canonical support-triage ticket corpus.

Output: 50 tickets across 10 clusters of 5 paraphrastic variants each.
Each cluster shares a category and a theme; the 5 variants within a
cluster express the same complaint in different words/voices so the
semantic cache has something meaningful to hit.

The script writes the corpus to `benchmarks/data/support_tickets.jsonl`
and the JSONL is the canonical input for benchmark runs. Re-run only
when intentionally regenerating; commit the new output as a single
diff.

Cost: ~$0.30 of Sonnet API at 2026-05 prices. Idempotent against
deterministic temperature=0 generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

BENCH_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BENCH_DIR / "data" / "support_tickets.jsonl"


# 10 clusters × 5 variants each = 50 tickets. Each cluster represents
# a recurring complaint pattern. Categories match support_triage.CATEGORIES.
CLUSTERS: list[dict] = [
    {
        "cluster_id": "subscription_double_charge",
        "category": "billing",
        "theme": "Customer was charged twice for their monthly subscription this billing cycle.",
    },
    {
        "cluster_id": "password_reset_loop",
        "category": "account",
        "theme": "Password reset link expires before the customer can click it, leaving them locked out.",
    },
    {
        "cluster_id": "package_stuck_in_transit",
        "category": "shipping",
        "theme": "Tracking info shows the package has been stuck at the same status for over a week.",
    },
    {
        "cluster_id": "app_crashes_on_launch",
        "category": "technical",
        "theme": "Customer's app crashes immediately on launch after the latest update.",
    },
    {
        "cluster_id": "wrong_item_received",
        "category": "shipping",
        "theme": "Customer received a completely different item than what they ordered.",
    },
    {
        "cluster_id": "refund_not_received",
        "category": "refund",
        "theme": "Customer was promised a refund 2-3 weeks ago and it hasn't arrived.",
    },
    {
        "cluster_id": "service_outage_now",
        "category": "outage",
        "theme": "Customer reports the service is currently completely down for them.",
    },
    {
        "cluster_id": "feature_request_export",
        "category": "feature_request",
        "theme": "Customer wants a CSV/PDF export feature for their data.",
    },
    {
        "cluster_id": "suspicious_login_alert",
        "category": "security",
        "theme": "Customer got an email about a suspicious login from an unfamiliar location.",
    },
    {
        "cluster_id": "positive_feedback_thanks",
        "category": "feedback",
        "theme": "Customer is writing in just to say they love the product and thank the team.",
    },
]


SYSTEM_PROMPT = """\
You generate realistic customer-support tickets for a benchmark dataset.
Given a complaint theme, write exactly 5 paraphrastic variants of the
same underlying complaint. The variants should:

- Express the same complaint with different word choices, sentence
  structures, levels of formality, and emotional tones.
- Read like real tickets — first person, 2-4 sentences, no greetings
  or signatures.
- Differ noticeably from each other so the dataset isn't trivially
  duplicate, but share enough core content that a semantic-similarity
  system would recognize them as related.

Output: a JSON array of exactly 5 strings, each one a ticket body.
No surrounding text, no markdown fence — just the JSON array.
"""


def synthesize_cluster(litellm_module, cluster: dict, model: str) -> list[str]:
    """Call Sonnet to produce 5 paraphrastic variants for one cluster."""
    response = litellm_module.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Complaint theme:\n{cluster['theme']}\n\n"
                    f"Generate 5 variants now."
                ),
            },
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    # Strip optional ```json fences if the model added them.
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    variants = json.loads(text)
    if not isinstance(variants, list) or len(variants) != 5:
        raise RuntimeError(
            f"Expected list of 5 strings for {cluster['cluster_id']}, got: {variants!r}"
        )
    return [str(v).strip() for v in variants]


def main() -> int:
    parser = argparse.ArgumentParser(prog="support_tickets_gen")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model used for synthesis (Sonnet by default).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    load_dotenv(BENCH_DIR / ".env")

    # Import litellm lazily so this script is importable without it.
    import litellm  # noqa: F401

    tickets: list[dict] = []
    next_id = 1
    for cluster in CLUSTERS:
        print(f"  cluster: {cluster['cluster_id']}", file=sys.stderr)
        variants = synthesize_cluster(litellm, cluster, args.model)
        for variant in variants:
            tickets.append(
                {
                    "id": f"T{next_id:03d}",
                    "category": cluster["category"],
                    "cluster": cluster["cluster_id"],
                    "text": variant,
                }
            )
            next_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ticket in tickets:
            f.write(json.dumps(ticket) + "\n")

    print(f"wrote {args.out} ({len(tickets)} tickets across {len(CLUSTERS)} clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
