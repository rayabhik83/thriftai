"""
`make judge` entry point — walk every `raw/<run_id>/artifacts.jsonl`
under results/, score with Opus, write `raw/<run_id>/judge_scores.jsonl`.

Designed to be idempotent: the judge module caches by artifact hash,
so re-running this script doesn't re-spend on artifacts already judged.

This is a separate step from the workload runner because:

- Judging costs real money (Opus pricing is ~60x Haiku per token).
- Judging is workload-independent — you can judge a finished smoke
  without re-running it.
- Quality numbers can lag the cost/latency numbers without blocking
  pipeline iteration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmarks.judge import llm_judge

BENCH_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BENCH_DIR / "results" / "raw"


_RUN_ID_RE = re.compile(
    r"\d{8}_\d{6}_(?P<workload>[a-z_]+?)_"
    r"(?P<condition>baseline|thriftai_cold|thriftai_warm|thriftai_replay)_"
    r"(?P<model>[a-z0-9\-]+)_seed(?P<seed>\d+)"
)


def _parse_run_id(run_id: str) -> dict[str, str] | None:
    """Reverse the format set in run._make_run_id."""
    m = _RUN_ID_RE.fullmatch(run_id)
    if m is None:
        return None
    return m.groupdict()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchmarks.runner.judge_runner")
    p.add_argument(
        "--budget",
        type=float,
        default=15.0,
        help="Hard cap (USD) checked against the shared spend ledger.",
    )
    args = p.parse_args(argv)

    load_dotenv(BENCH_DIR / ".env")

    if not RAW_DIR.exists():
        print(f"no raw results found at {RAW_DIR}; nothing to judge.", file=sys.stderr)
        return 0

    # Sanity check against the ledger before doing anything.
    from benchmarks.runner import budget as _budget

    before = _budget.total_spent()
    print(f"budget: spent so far ${before:.4f}, cap ${args.budget:.2f}")

    cells = sorted(RAW_DIR.glob("*/artifacts.jsonl"))
    if not cells:
        print("no artifacts.jsonl files in raw/ — workload runs need to save them.")
        return 0

    for artifacts_path in cells:
        run_dir = artifacts_path.parent
        meta = _parse_run_id(run_dir.name)
        if meta is None:
            print(f"  SKIP (unparseable run_id): {run_dir.name}")
            continue
        workload = meta["workload"]
        out_path = run_dir / "judge_scores.jsonl"

        if workload not in llm_judge.RUBRICS:
            print(f"  SKIP (no rubric for workload={workload}): {run_dir.name}")
            continue

        print(f"  judging: {run_dir.name}")
        sys.stdout.flush()
        scores = llm_judge.judge_artifacts_file(artifacts_path, workload)

        # Cap check after each cell so we abort cleanly mid-stream.
        spent = _budget.total_spent()
        if spent > args.budget:
            print(
                f"BUDGET EXCEEDED: spent ${spent:.4f} > cap ${args.budget:.2f}. "
                f"Stopping before next cell.",
                file=sys.stderr,
            )
            return 1

        with out_path.open("w") as f:
            for s in scores:
                f.write(json.dumps(s) + "\n")

    after = _budget.total_spent()
    print(f"budget: judged in this run ${after - before:.4f}, total ${after:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
