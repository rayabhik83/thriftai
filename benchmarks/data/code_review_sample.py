"""
Sample the canonical code-review corpus from a public Python code dataset.

Originally targeted `bigcode/the-stack-smol-xs` per the brief, but that
dataset is gated (requires HF auth + signed terms) and uses the old
dataset-script format that current `datasets` no longer supports. We
switched to `codeparrot/codeparrot-clean-valid` which is non-gated,
Python-only, and exposes the same content/license/path shape. Both
datasets pull from public OSS code with permissive licenses, so the
character of the benchmark inputs is materially the same.

Output: 20 Python snippets, each a reasonably-sized function or short
module. Both the sampling script and the resulting JSONL are committed;
the JSONL is the canonical input for the workload.

Idempotent: we hash-sort entries that pass the size filter and take
the first 20, so re-running produces the same output (modulo upstream
dataset changes).

Cost: \$0 API. ~100 MB transient disk for the HF dataset cache.
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


def sample(out_path: Path, n: int = 20, dataset_id: str = "codeparrot/codeparrot-clean-valid") -> int:
    """Pull `n` snippets from the dataset, return count written."""
    from datasets import load_dataset

    print(f"  loading {dataset_id} (streaming)...", file=sys.stderr)
    sys.stderr.flush()
    # codeparrot/codeparrot-clean-valid is the validation slice of
    # codeparrot-clean: Python-only, license-tagged, no auth required.
    # We use streaming mode to avoid downloading the entire dataset.
    ds = load_dataset(dataset_id, split="train", streaming=True)

    selected: list[dict] = []
    inspected = 0
    # Iterate in stable order (hash of content) so we always pick the
    # same N entries for the same dataset snapshot. Cap inspection at
    # ~5000 rows so streaming doesn't run forever.
    for row in ds:
        inspected += 1
        if inspected > 5000:
            break
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
    parser.add_argument("--dataset", default="codeparrot/codeparrot-clean-valid")
    args = parser.parse_args()

    n = sample(args.out, n=args.n, dataset_id=args.dataset)
    print(f"wrote {args.out} ({n} snippets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
