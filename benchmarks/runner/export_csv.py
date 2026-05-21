"""
Dump the plot-source aggregates to CSV so they survive `make clean` and
can be loaded into any tool (Datawrapper, R/ggplot, Observable, Excel, etc).

Outputs three CSVs under `benchmarks/results/`:

- aggregates_cost.csv     one row per (workload, condition, model, seed, task_id)
                          with paid_usd and would_have_usd. Feeds the
                          cost-reduction plot and the quality-cost Pareto.
- aggregates_quality.csv  one row per (workload, condition, model, task_id)
                          with the mean rubric score (1-5) or pass@1.
- aggregates_latency.csv  one row per (workload, condition, model, latency_ms,
                          broker_resolution). Long-format, ready for ggplot
                          / Seaborn / Vega-Lite.

These are the exact tuples the report's matplotlib code averages over.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BENCH_DIR / "results" / "raw"
PRICING_PATH = BENCH_DIR / "pricing.yaml"
OUT_DIR = BENCH_DIR / "results"

_RUN_ID_RE = re.compile(
    r"\d{8}_\d{6}_(?P<workload>[a-z_]+?)_"
    r"(?P<condition>baseline|thriftai_cold|thriftai_warm|thriftai_replay)_"
    r"(?P<model>[a-z0-9\-]+)_seed(?P<seed>\d+)"
)


def _load_pricing() -> dict:
    import yaml
    with PRICING_PATH.open() as f:
        return yaml.safe_load(f).get("models", {})


def _cost(model: str, in_tok: int, out_tok: int, pricing: dict) -> float:
    entry = pricing.get(model)
    if entry is None:
        return 0.0
    return (in_tok / 1_000_000) * entry["input_per_million_usd"] + (
        out_tok / 1_000_000
    ) * entry["output_per_million_usd"]


def export_cost_csv(out_path: Path, pricing: dict) -> int:
    """One row per (run, task). paid_usd + would_have_usd, both pricing-derived."""
    paid: dict[tuple, float] = defaultdict(float)
    would: dict[tuple, float] = defaultdict(float)
    meta: dict[tuple, tuple] = {}

    if not RAW_DIR.exists():
        return 0
    for jsonl in RAW_DIR.glob("*/calls.jsonl"):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                c = _cost(r["model"], r["input_tokens"], r["output_tokens"], pricing)
                key = (
                    r["run_id"], r["workload"], r["condition"],
                    r["model_under_test"], r["seed"], r["task_id"],
                )
                meta[key] = (r["workload"], r["condition"], r["model_under_test"], r["seed"], r["task_id"])
                would[key] += c
                if r["broker_resolution"] == "live":
                    paid[key] += c

    rows = []
    for key, (workload, condition, model, seed, task_id) in meta.items():
        rows.append({
            "workload": workload,
            "condition": condition,
            "model": model,
            "seed": seed,
            "task_id": task_id,
            "paid_usd": round(paid.get(key, 0.0), 8),
            "would_have_usd": round(would[key], 8),
            "saved_usd": round(would[key] - paid.get(key, 0.0), 8),
        })
    rows.sort(key=lambda r: (r["workload"], r["model"], r["condition"], r["seed"], r["task_id"]))

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
            ["workload", "condition", "model", "seed", "task_id", "paid_usd", "would_have_usd", "saved_usd"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_quality_csv(out_path: Path) -> int:
    """One row per (cell, task). score: mean of rubric ints (1-5) or pass@1×4+1."""
    rows = []
    if not RAW_DIR.exists():
        return 0
    for run_dir in sorted(RAW_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _RUN_ID_RE.fullmatch(run_dir.name)
        if m is None:
            continue
        workload, condition, model, seed = (
            m["workload"], m["condition"], m["model"], int(m["seed"])
        )

        judge_path = run_dir / "judge_scores.jsonl"
        if judge_path.exists():
            with judge_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    int_fields = [v for k, v in row.items() if isinstance(v, int)]
                    if int_fields:
                        rows.append({
                            "workload": workload,
                            "condition": condition,
                            "model": model,
                            "seed": seed,
                            "task_id": row.get("task_id", ""),
                            "score": round(sum(int_fields) / len(int_fields), 4),
                            "source": "judge",
                        })

        humaneval_path = run_dir / "humaneval_scores.jsonl"
        if humaneval_path.exists():
            with humaneval_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    p = row.get("pass_at_1")
                    if p is not None:
                        rows.append({
                            "workload": workload,
                            "condition": condition,
                            "model": model,
                            "seed": seed,
                            "task_id": row.get("task_id", ""),
                            "score": 1.0 + 4.0 * float(p),
                            "source": "pass@1",
                        })

    rows.sort(key=lambda r: (r["workload"], r["model"], r["condition"], r["seed"], r["task_id"]))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["workload", "condition", "model", "seed", "task_id", "score", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_latency_csv(out_path: Path) -> int:
    """One row per individual call. Long format for distribution plots."""
    rows = []
    if not RAW_DIR.exists():
        return 0
    for jsonl in RAW_DIR.glob("*/calls.jsonl"):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append({
                    "workload": r["workload"],
                    "condition": r["condition"],
                    "model": r["model_under_test"],
                    "seed": r["seed"],
                    "agent_name": r["agent_name"],
                    "broker_resolution": r["broker_resolution"],
                    "latency_total_ms": r["latency_total_ms"],
                    "latency_api_ms": r.get("latency_api_ms") if r.get("latency_api_ms") is not None else "",
                    "latency_overhead_ms": r["latency_overhead_ms"],
                })
    rows.sort(key=lambda r: (r["workload"], r["model"], r["condition"], r["seed"]))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
            ["workload", "condition", "model", "seed", "agent_name", "broker_resolution",
             "latency_total_ms", "latency_api_ms", "latency_overhead_ms"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pricing = _load_pricing()

    n_cost = export_cost_csv(OUT_DIR / "aggregates_cost.csv", pricing)
    n_qual = export_quality_csv(OUT_DIR / "aggregates_quality.csv")
    n_lat = export_latency_csv(OUT_DIR / "aggregates_latency.csv")

    print(f"wrote aggregates_cost.csv    ({n_cost} rows)")
    print(f"wrote aggregates_quality.csv ({n_qual} rows)")
    print(f"wrote aggregates_latency.csv ({n_lat} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
