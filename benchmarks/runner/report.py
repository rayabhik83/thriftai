"""
Render REPORT.md from raw JSONL logs in benchmarks/results/raw/ +
benchmarks/pricing.yaml.

This module is the only thing that writes REPORT.md. It's pure:

  raw JSONL + pricing.yaml → REPORT.md + plots/

No API calls, idempotent, safe to re-run anytime. Designed so that
`make report` can regenerate the published numbers from a checked-in
results/raw/ tree (which we'll only commit selectively — see PLAN.md).

When there is no data yet (empty results/raw/), the report still
renders with section headers and explicit "no data yet" placeholders
so the structure is reviewable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import stats

# Resolve from the file location so this works regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"
RAW_DIR = BENCH_DIR / "results" / "raw"
REPORT_PATH = BENCH_DIR / "results" / "REPORT.md"
PRICING_PATH = BENCH_DIR / "pricing.yaml"


# ---- IO -------------------------------------------------------------------


def load_pricing(path: Path = PRICING_PATH) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_calls(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load every JSONL record under results/raw/. Empty list if nothing exists."""
    records: list[dict[str, Any]] = []
    if not raw_dir.exists():
        return records
    for jsonl in sorted(raw_dir.glob("*/calls.jsonl")):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


# ---- Cost recomputation from pricing.yaml ---------------------------------


def cost_from_pricing(
    model: str, input_tokens: int, output_tokens: int, pricing: dict[str, Any]
) -> float | None:
    """Canonical $ figure for the report. None if model not in pricing.yaml."""
    entry = pricing.get("models", {}).get(model)
    if entry is None:
        return None
    return (
        input_tokens / 1_000_000.0 * entry["input_per_million_usd"]
        + output_tokens / 1_000_000.0 * entry["output_per_million_usd"]
    )


# ---- Aggregation ----------------------------------------------------------


def _per_task_cost(
    records: list[dict[str, Any]], pricing: dict[str, Any]
) -> tuple[dict, dict]:
    """Group by (workload, condition, model_under_test) → (paid, would-have) costs.

    Returns (paid_by_cell, would_have_by_cell). Each maps a cell to a list
    of per-task totals (one entry per task per seed).

    - **paid**: actual USD spent. Cache hits and replays count as $0;
      only `live` resolutions cost money.
    - **would-have**: cost computed from token counts as if every call
      had gone live. Used to compute savings (would_have - paid).

    Costs are computed from `pricing.yaml`, not from any LiteLLM-reported
    number, so the figures are recomputable from raw logs.
    """
    paid_by_task: dict[tuple, float] = defaultdict(float)
    would_have_by_task: dict[tuple, float] = defaultdict(float)
    cell_keys: dict[tuple, tuple] = {}

    for r in records:
        cost = cost_from_pricing(
            r["model"], r["input_tokens"], r["output_tokens"], pricing
        )
        if cost is None:
            continue
        task_key = (
            r["run_id"], r["workload"], r["condition"],
            r["model_under_test"], r["seed"], r["task_id"],
        )
        cell_keys[task_key] = (r["workload"], r["condition"], r["model_under_test"])
        would_have_by_task[task_key] += cost
        if r["broker_resolution"] == "live":
            paid_by_task[task_key] += cost
        # else: cache_hit / semantic_hit / replay → paid is $0 for this call

    paid_by_cell: dict[tuple, list[float]] = defaultdict(list)
    would_have_by_cell: dict[tuple, list[float]] = defaultdict(list)
    for task_key, cell in cell_keys.items():
        paid_by_cell[cell].append(paid_by_task.get(task_key, 0.0))
        would_have_by_cell[cell].append(would_have_by_task[task_key])
    return dict(paid_by_cell), dict(would_have_by_cell)


def _per_call_latency(records: list[dict[str, Any]]) -> dict[tuple, list[float]]:
    """Group by cell → list of per-call total latency ms."""
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        cell = (r["workload"], r["condition"], r["model_under_test"])
        latency = r.get("latency_total_ms")
        if latency is not None:
            by_cell[cell].append(float(latency))
    return dict(by_cell)


def _resolution_counts(records: list[dict[str, Any]]) -> dict[tuple, dict[str, int]]:
    """Per cell → resolution name → count."""
    by_cell: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        cell = (r["workload"], r["condition"], r["model_under_test"])
        by_cell[cell][r["broker_resolution"]] += 1
    return {k: dict(v) for k, v in by_cell.items()}


# ---- Rendering ------------------------------------------------------------


def _render_headline(
    paid_by_cell: dict, would_have_by_cell: dict, latency_by_cell: dict
) -> str:
    """Headline table: workload × condition × model → $/task paid, $/task saved, latency."""
    if not paid_by_cell and not latency_by_cell:
        return (
            "| Workload | Condition | Model | $/task paid (mean ± std) | "
            "$/task saved | p50 latency (ms) | p95 latency (ms) |\n"
            "|---|---|---|---|---|---|---|\n"
            "| _no data yet_ |   |   |   |   |   |   |\n"
        )

    lines = [
        "| Workload | Condition | Model | $/task paid (mean ± std) | "
        "$/task saved (mean ± std) | p50 latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    cells = sorted(
        set(paid_by_cell.keys())
        | set(would_have_by_cell.keys())
        | set(latency_by_cell.keys())
    )
    for cell in cells:
        workload, condition, model = cell
        paid = paid_by_cell.get(cell, [])
        would_have = would_have_by_cell.get(cell, [])
        # Per-task savings = would-have - paid, pairwise.
        savings = (
            [w - p for w, p in zip(would_have, paid)]
            if len(paid) == len(would_have)
            else []
        )
        paid_cell = stats.fmt_mean_std(paid, precision=4, unit=" $")
        saved_cell = stats.fmt_mean_std(savings, precision=4, unit=" $")
        latencies = latency_by_cell.get(cell, [])
        p50 = f"{stats.p50(latencies):.0f}" if latencies else "—"
        p95 = f"{stats.p95(latencies):.0f}" if latencies else "—"
        lines.append(
            f"| {workload} | {condition} | {model} | {paid_cell} | {saved_cell} | "
            f"{p50} | {p95} |"
        )
    return "\n".join(lines) + "\n"


def _render_resolution_breakdown(counts: dict) -> str:
    if not counts:
        return "_no data yet_\n"
    lines = [
        "| Workload | Condition | Model | live | cache_hit | semantic_hit | replay |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in sorted(counts.keys()):
        workload, condition, model = cell
        cs = counts[cell]
        lines.append(
            f"| {workload} | {condition} | {model} | "
            f"{cs.get('live', 0)} | {cs.get('cache_hit', 0)} | "
            f"{cs.get('semantic_hit', 0)} | {cs.get('replay', 0)} |"
        )
    return "\n".join(lines) + "\n"


def render(records: list[dict[str, Any]], pricing: dict[str, Any]) -> str:
    """Build the full report markdown string. Pure function — no IO."""
    paid_by_cell, would_have_by_cell = _per_task_cost(records, pricing)
    latency_by_cell = _per_call_latency(records)
    counts = _resolution_counts(records)

    pulled_on = pricing.get("pulled_on", "unknown")
    source_url = pricing.get("source_url", "")
    n_records = len(records)
    n_runs = len({r["run_id"] for r in records}) if records else 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"# ThriftAI Benchmark Results\n\n"
        f"> Generated {now} from {n_records} calls across {n_runs} run(s).\n"
        f"> Pricing snapshot: pulled {pulled_on}"
        + (f" — [source]({source_url})" if source_url else "")
        + ".\n\n"
        "## Headline\n\n"
        + _render_headline(paid_by_cell, would_have_by_cell, latency_by_cell)
        + "\n## Call resolution breakdown\n\n"
        "Counts of brokered-call outcomes per cell. Cache vs replay vs live\n"
        "tells you which mechanism is doing the work.\n\n"
        + _render_resolution_breakdown(counts)
        + "\n## Per-workload deep dives\n\n"
        "_filled in once workloads land._\n\n"
        "## Methodology\n\n"
        "See `benchmarks/README.md` and `benchmarks/PLAN.md`.\n\n"
        "## Raw data\n\n"
        "Per-call records under `benchmarks/results/raw/<run_id>/calls.jsonl`.\n"
        "Every dollar figure above is derived from raw token counts in those\n"
        "files multiplied by `benchmarks/pricing.yaml`; see `make rederive`\n"
        "for the verification script.\n"
    )


def write_report(content: str, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def main() -> None:
    # Resolve paths from the module attributes at call time so tests can
    # monkeypatch them. Don't rely on the function-default values, which
    # are bound at definition time.
    pricing = load_pricing(PRICING_PATH)
    records = load_calls(RAW_DIR)
    content = render(records, pricing)
    out = write_report(content, REPORT_PATH)
    print(f"wrote {out} ({len(content)} bytes; {len(records)} call records)")


if __name__ == "__main__":
    main()
