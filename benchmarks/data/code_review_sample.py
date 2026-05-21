"""
Sample the canonical code-review corpus from `bigcode/the-stack-smol`.

Output: 20 Python snippets, each one a reasonably-sized function or
short module that gives a code reviewer enough to chew on without
being so large that the reviewer's response would blow our token
budget. Both the sampling script and the resulting JSONL are
committed; the JSONL is the canonical input for the workload.

Idempotent: we hash-sort the dataset and take the first 20 entries
that pass the size filter, so re-running produces the same output
(modulo upstream dataset changes).

Cost: \$0 API. ~50-200 MB transient disk for the HF dataset cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BENCH_DIR / "data" / "code_review_snippets.jsonl"


# Sized so a typical reviewer pass reads in a few hundred input tokens
# and emits a few hundred output tokens — fits comfortably under the
# per-minute rate-limit math at Haiku rates.
MIN_LINES = 8
MAX_LINES = 40
MIN_CHARS = 200
MAX_CHARS = 2000


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sample(out_path: Path, n: int = 20, dataset_id: str = "bigcode/the-stack-smol") -> int:
    """Pull `n` snippets from the dataset, return count written."""
    from datasets import load_dataset

    print(f"  loading {dataset_id} (Python config)...", file=sys.stderr)
    sys.stderr.flush()
    # bigcode/the-stack-smol exposes one config per language. Python is
    # the natural choice for code review samples.
    ds = load_dataset(dataset_id, data_dir="data/python", split="train", streaming=False)

    selected: list[dict] = []
    # Iterate in stable order (hash of content) so we always pick the
    # same N entries for the same dataset snapshot.
    for row in ds:
        content = row.get("content", "")
        if not isinstance(content, str):
            continue
        n_lines = content.count("\n") + 1
        n_chars = len(content)
        if not (MIN_LINES <= n_lines <= MAX_LINES):
            continue
        if not (MIN_CHARS <= n_chars <= MAX_CHARS):
            continue
        # Skip files that are mostly imports or boilerplate.
        non_import_lines = [
            ln for ln in content.splitlines()
            if ln.strip() and not ln.strip().startswith(("import ", "from "))
        ]
        if len(non_import_lines) < MIN_LINES:
            continue
        selected.append(
            {
                "content": content,
                "path": row.get("path", ""),
                "license": row.get("license", ""),
                "hash": _stable_hash(content),
            }
        )

    # Sort by hash for determinism, then take the first n.
    selected.sort(key=lambda r: r["hash"])
    selected = selected[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, r in enumerate(selected, start=1):
            f.write(
                json.dumps(
                    {
                        "id": f"C{i:03d}",
                        "language": "python",
                        "path": r["path"],
                        "license": r["license"],
                        "code": r["content"],
                    }
                )
                + "\n"
            )
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(prog="code_review_sample")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dataset", default="bigcode/the-stack-smol")
    args = parser.parse_args()

    n = sample(args.out, n=args.n, dataset_id=args.dataset)
    print(f"wrote {args.out} ({n} snippets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
