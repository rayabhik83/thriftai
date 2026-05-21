"""
Generate the three required figures for REPORT.md:

  1. cost_vs_repetitiveness.png
     Cost reduction (1 - paid/would_have) per condition × workload.
     Workloads ordered left-to-right by intrinsic repetitiveness
     (support_triage clusters most heavily; humaneval least).

  2. quality_cost_pareto.png
     Per-cell scatter: x = mean $/task paid, y = mean judge quality.
     Connected lines for the same (workload, model) across conditions
     so the cost-vs-quality trade-off shows.

  3. latency_distribution.png
     Per-cell box/violin: latency_total_ms distributions per condition.
     Baseline vs thriftai_cold should overlap; thriftai_warm should
     be dramatically lower.

All plots are saved as PNG under benchmarks/results/plots/ and are
re-embeddable from REPORT.md.

Inputs: same raw JSONL the report module reads + optional
judge_scores.jsonl when present.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so this works in CI
import matplotlib.pyplot as plt

from .report import RAW_DIR, cost_from_pricing, load_calls, load_pricing

BENCH_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BENCH_DIR / "results" / "plots"


# ---- shared helpers -------------------------------------------------------


def _per_cell_aggregates(
    records: list[dict[str, Any]], pricing: dict[str, Any]
) -> dict[tuple, dict]:
    """For each (workload, condition, model) cell, compute:
    - mean $/task paid
    - mean $/task would_have
    - all latency_total_ms values
    - resolution counts
    """
    by_task_paid: dict[tuple, float] = defaultdict(float)
    by_task_would: dict[tuple, float] = defaultdict(float)
    cell_keys: dict[tuple, tuple] = {}
    by_cell_lat: dict[tuple, list[float]] = defaultdict(list)
    by_cell_res: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        cost = cost_from_pricing(r["model"], r["input_tokens"], r["output_tokens"], pricing)
        if cost is None:
            continue
        task_key = (
            r["run_id"], r["workload"], r["condition"],
            r["model_under_test"], r["seed"], r["task_id"],
        )
        cell_key = (r["workload"], r["condition"], r["model_under_test"])
        cell_keys[task_key] = cell_key
        by_task_would[task_key] += cost
        if r["broker_resolution"] == "live":
            by_task_paid[task_key] += cost
        by_cell_lat[cell_key].append(float(r.get("latency_total_ms", 0)))
        by_cell_res[cell_key][r["broker_resolution"]] += 1

    by_cell: dict[tuple, dict] = {}
    for task_key, cell_key in cell_keys.items():
        agg = by_cell.setdefault(cell_key, {"paid": [], "would": [], "lat": [], "res": {}})
        agg["paid"].append(by_task_paid.get(task_key, 0.0))
        agg["would"].append(by_task_would[task_key])
    for cell_key, lat in by_cell_lat.items():
        by_cell.setdefault(cell_key, {})["lat"] = lat
    for cell_key, res in by_cell_res.items():
        by_cell.setdefault(cell_key, {})["res"] = dict(res)
    return by_cell


def _load_judge_scores() -> dict[tuple, list[float]]:
    """For each cell, return list of per-task aggregate quality scores.

    Reads benchmarks/results/raw/<run_id>/judge_scores.jsonl. Each line
    has the three 1-5 fields; we average them per task to a 1-5 overall.
    """
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    if not RAW_DIR.exists():
        return {}
    for run_dir in sorted(RAW_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        scores_path = run_dir / "judge_scores.jsonl"
        if not scores_path.exists():
            continue
        meta = _parse_run_dir_name(run_dir.name)
        if meta is None:
            continue
        cell_key = (meta["workload"], meta["condition"], meta["model"])
        with scores_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fields = [v for k, v in row.items() if isinstance(v, int)]
                if fields:
                    by_cell[cell_key].append(sum(fields) / len(fields))
    return dict(by_cell)


def _parse_run_dir_name(name: str) -> dict[str, str] | None:
    """Pull (workload, condition, model, seed) back out of a run-id dir name."""
    import re
    m = re.fullmatch(
        r"\d{8}_\d{6}_(?P<workload>[a-z_]+?)_"
        r"(?P<condition>baseline|thriftai_cold|thriftai_warm|thriftai_replay)_"
        r"(?P<model>[a-z0-9\-]+)_seed(?P<seed>\d+)",
        name,
    )
    return m.groupdict() if m else None


# ---- the three figures ----------------------------------------------------


def plot_cost_reduction(by_cell: dict[tuple, dict], out_path: Path) -> Path:
    """Bar chart: cost reduction percent per condition, grouped by workload."""
    # Aggregate to (workload, condition) means.
    by_wc: dict[tuple, list[float]] = defaultdict(list)
    for (workload, condition, _model), agg in by_cell.items():
        if not agg.get("paid") or not agg.get("would"):
            continue
        paid_total = sum(agg["paid"])
        would_total = sum(agg["would"])
        if would_total == 0:
            continue
        savings_pct = 100.0 * (1.0 - paid_total / would_total)
        by_wc[(workload, condition)].append(savings_pct)

    if not by_wc:
        return out_path  # nothing to plot

    workloads = sorted({w for (w, _c) in by_wc.keys()})
    conditions = ["baseline", "thriftai_cold", "thriftai_warm"]
    bar_data = {
        c: [
            sum(by_wc.get((w, c), [0.0])) / max(len(by_wc.get((w, c), [0.0])), 1)
            for w in workloads
        ]
        for c in conditions
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(workloads))
    width = 0.25
    for i, c in enumerate(conditions):
        ax.bar(
            [xi + i * width for xi in x],
            bar_data[c],
            width=width,
            label=c,
        )
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Cost reduction vs. baseline (%)")
    ax.set_title("Cost reduction by workload × condition")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_quality_cost_pareto(
    by_cell: dict[tuple, dict],
    judge_by_cell: dict[tuple, list[float]],
    out_path: Path,
) -> Path:
    """Scatter: mean $/task paid (x) vs mean judge score (y), per cell.

    Lines connect points sharing (workload, model) so the user can see
    the trajectory baseline → cold → warm.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    grouped: dict[tuple, list[tuple[str, float, float]]] = defaultdict(list)
    for (workload, condition, model), agg in by_cell.items():
        paid = sum(agg.get("paid", [])) / max(len(agg.get("paid", [])), 1)
        quality_vals = judge_by_cell.get((workload, condition, model), [])
        quality = sum(quality_vals) / len(quality_vals) if quality_vals else None
        if quality is None:
            continue
        grouped[(workload, model)].append((condition, paid, quality))

    for (workload, model), points in grouped.items():
        order = {"baseline": 0, "thriftai_cold": 1, "thriftai_warm": 2}
        points.sort(key=lambda p: order.get(p[0], 99))
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, marker="o", label=f"{workload} / {model}")
        for cond, x, y in points:
            ax.annotate(cond, (x, y), fontsize=8, alpha=0.7)

    ax.set_xlabel("Mean $/task paid (USD)")
    ax.set_ylabel("Mean judge score (1-5)")
    ax.set_title("Quality vs. cost — Pareto trajectory per workload")
    if grouped:
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_latency_distribution(by_cell: dict[tuple, dict], out_path: Path) -> Path:
    """Box plot of per-call latency_total_ms, grouped by condition."""
    by_cond: dict[str, list[float]] = defaultdict(list)
    for (_workload, condition, _model), agg in by_cell.items():
        by_cond[condition].extend(agg.get("lat", []))

    if not by_cond:
        return out_path

    conditions = sorted(by_cond.keys(), key=lambda c: (
        0 if c == "baseline" else 1 if c == "thriftai_cold" else 2
    ))
    data = [by_cond[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=conditions, showfliers=False)
    ax.set_ylabel("Latency per call (ms)")
    ax.set_title("Latency distribution by condition (all workloads, all models)")
    ax.set_yscale("symlog")  # warm cache → 1ms, baseline → 1000ms; symlog reads better
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---- entry point ----------------------------------------------------------


def main() -> int:
    pricing = load_pricing()
    records = load_calls()
    if not records:
        print("plots: no raw records under benchmarks/results/raw/; nothing to plot.")
        return 0

    by_cell = _per_cell_aggregates(records, pricing)
    judge_by_cell = _load_judge_scores()

    plot_cost_reduction(by_cell, PLOTS_DIR / "cost_reduction.png")
    plot_quality_cost_pareto(by_cell, judge_by_cell, PLOTS_DIR / "quality_cost_pareto.png")
    plot_latency_distribution(by_cell, PLOTS_DIR / "latency_distribution.png")
    print(f"wrote 3 figures to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
