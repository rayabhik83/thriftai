"""
Persistent spend tracker for the benchmark suite.

Every `live` resolution recorded by the instrumentation appends one line
to `benchmarks/cache/spend_ledger.jsonl`. The runner reads the ledger
on each call and aborts if the cumulative spend exceeds the cap passed
via `--budget`.

The ledger is **persistent across runs**: if a user runs `make smoke`
twice, the second run sees the spend from the first. This matches the
user's intent ("we don't spend more than $10") more cleanly than a
per-invocation budget.

Reset with: `python -m benchmarks.runner.budget reset`
Inspect with: `python -m benchmarks.runner.budget total`
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
LEDGER_PATH = BENCH_DIR / "cache" / "spend_ledger.jsonl"

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """Raised when total spend exceeds the configured cap."""


def total_spent() -> float:
    """Sum of all amounts recorded to the ledger."""
    if not LEDGER_PATH.exists():
        return 0.0
    total = 0.0
    with LEDGER_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line).get("amount_usd", 0.0))
            except (ValueError, json.JSONDecodeError):
                # Skip malformed lines rather than crash a long run.
                continue
    return total


def record(amount_usd: float, *, run_id: str, condition: str | None = None) -> float:
    """Append a spend record to the ledger and return the new total."""
    if amount_usd <= 0:
        return total_spent()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with LEDGER_PATH.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "amount_usd": float(amount_usd),
                        "run_id": run_id,
                        "condition": condition,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
    return total_spent()


def check(cap_usd: float) -> None:
    """Raise BudgetExceeded if total recorded spend exceeds cap_usd."""
    spent = total_spent()
    if spent > cap_usd:
        raise BudgetExceeded(
            f"Spend ${spent:.4f} exceeds budget cap ${cap_usd:.2f}. "
            f"Raise the cap on the next invocation with --budget, or "
            f"reset the ledger: python -m benchmarks.runner.budget reset"
        )


def reset() -> None:
    """Clear the persistent ledger."""
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.runner.budget")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("total", help="Print total recorded spend in USD.")
    sub.add_parser("reset", help="Clear the spend ledger.")
    args = parser.parse_args()

    if args.cmd == "total":
        print(f"${total_spent():.4f}")
        return 0
    if args.cmd == "reset":
        reset()
        print("ledger cleared")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
