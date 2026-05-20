"""
Verification script: recompute every dollar figure in the report from raw
JSONL + pricing.yaml, and compare against what litellm reported.

The point is to give a skeptical reader a way to confirm that the published
numbers can be derived from raw data they can see. Exits non-zero if any
recomputed total disagrees with the report.

For now the script just totals per-cell costs from raw logs and prints
them. Once REPORT.md has concrete published numbers (post-workload-runs),
we'll add the diff-against-report step here.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from .report import (
    cost_from_pricing,
    load_calls,
    load_pricing,
)


def main() -> int:
    pricing = load_pricing()
    records = load_calls()
    if not records:
        print("rederive: no records under benchmarks/results/raw/; nothing to verify.")
        return 0

    # Sum recomputed cost per (workload, condition, model) cell and compare
    # to the litellm-reported cost summed the same way.
    recomputed: dict[tuple, float] = defaultdict(float)
    reported_litellm: dict[tuple, float] = defaultdict(float)

    skipped_unknown_model = 0
    for r in records:
        cell = (r["workload"], r["condition"], r["model_under_test"])
        c = cost_from_pricing(r["model"], r["input_tokens"], r["output_tokens"], pricing)
        if c is None:
            skipped_unknown_model += 1
            continue
        recomputed[cell] += c
        reported_litellm[cell] += float(r.get("actual_cost_usd_litellm") or 0.0)

    print(f"rederive: {len(records)} records, "
          f"{skipped_unknown_model} skipped (model not in pricing.yaml)")
    print()
    print(f"{'workload':<24}{'condition':<18}{'model':<24}"
          f"{'recomputed':>12}{'litellm':>12}{'delta':>10}")
    for cell in sorted(recomputed.keys()):
        wl, cond, model = cell
        a = recomputed[cell]
        b = reported_litellm[cell]
        delta_pct = (a - b) / b * 100 if b else 0.0
        print(f"{wl:<24}{cond:<18}{model:<24}"
              f"{a:>12.4f}{b:>12.4f}{delta_pct:>9.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
