"""
Score HumanEval completions saved by the runner.

For every run directory under `benchmarks/results/raw/` whose run_id
matches the humaneval workload, run the official `human_eval` evaluator
on the artifacts.jsonl, write a humaneval_scores.jsonl back into the
same directory, and emit pass@1 per cell.

This is a separate step from the workload runner because:
- Scoring requires unsandboxed Python execution of model output;
  best done as an explicit, opt-in step.
- Re-scoring an existing artifacts file is free — no API calls.

The report module's quality column auto-picks up pass@1 when it sees
humaneval_scores.jsonl (sidecar mirror of judge_scores.jsonl).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BENCH_DIR / "results" / "raw"


def _score_with_humaneval(artifacts: list[dict]) -> dict[str, float]:
    """Run human_eval's official evaluator on the saved completions.

    Returns a dict mapping task_id → 1.0 (passed) or 0.0 (failed).
    """
    from human_eval.evaluation import evaluate_functional_correctness

    # Convert our id format (HumanEval_0) back to upstream's (HumanEval/0).
    completions: list[dict] = []
    for a in artifacts:
        task_id = a.get("task_id", "")
        if "_" in task_id and not task_id.startswith("HumanEval/"):
            task_id = task_id.replace("_", "/", 1)
        completions.append({"task_id": task_id, "completion": a["completion"]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")
        path = f.name

    # `evaluate_functional_correctness` writes results to a sidecar file.
    results = evaluate_functional_correctness(
        sample_file=path,
        k=[1],
        n_workers=1,
        timeout=10.0,
        ignore_incomplete=True,
    )

    # Parse the sidecar file for per-task pass/fail.
    out: dict[str, float] = {}
    sidecar = path + "_results.jsonl"
    if Path(sidecar).exists():
        with open(sidecar) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tid = row["task_id"].replace("/", "_")
                out[tid] = 1.0 if row.get("passed") else 0.0
    return out


def main() -> int:
    if not RAW_DIR.exists():
        print("no raw results to score.", file=sys.stderr)
        return 0

    humaneval_cells = [
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and "_humaneval_" in d.name
    ]
    if not humaneval_cells:
        print("no humaneval cells found; run the workload first.", file=sys.stderr)
        return 0

    for cell_dir in sorted(humaneval_cells):
        artifacts_path = cell_dir / "artifacts.jsonl"
        if not artifacts_path.exists():
            continue
        with artifacts_path.open() as f:
            artifacts = [json.loads(line) for line in f if line.strip()]

        print(f"  scoring: {cell_dir.name}")
        sys.stdout.flush()
        per_task = _score_with_humaneval(artifacts)

        scores_path = cell_dir / "humaneval_scores.jsonl"
        with scores_path.open("w") as f:
            for tid, score in per_task.items():
                f.write(json.dumps({"task_id": tid, "pass_at_1": score}) + "\n")

        n = len(per_task)
        passed = int(sum(per_task.values()))
        print(f"    pass@1 = {passed}/{n} = {passed / n * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
