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
import re
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


_RUN_ID_RE = re.compile(
    r"\d{8}_\d{6}_(?P<workload>[a-z_]+?)_"
    r"(?P<condition>baseline|thriftai_cold|thriftai_warm|thriftai_replay)_"
    r"(?P<model>[a-z0-9\-]+)_seed(?P<seed>\d+)"
)


def _load_judge_scores(raw_dir: Path = RAW_DIR) -> dict[tuple, list[float]]:
    """Per cell → list of per-task aggregate quality scores.

    Two sources, both optional sidecars next to the artifacts:

    - judge_scores.jsonl: LLM-as-judge rubric scores (1-5 ints). We
      average all int fields per task and emit a 1-5 number.
    - humaneval_scores.jsonl: pass@1 per task (1.0 or 0.0). We use
      the pass@1 value directly and rescale to a 1-5 quality reading
      for the headline (1.0 → 5.0, 0.0 → 1.0) so the column is
      consistent across workloads.

    For a single cell at most one of the two files exists, so there's
    no double-counting.
    """
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    if not raw_dir.exists():
        return {}
    for run_dir in sorted(raw_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _RUN_ID_RE.fullmatch(run_dir.name)
        if m is None:
            continue
        cell = (m["workload"], m["condition"], m["model"])

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
                        by_cell[cell].append(sum(int_fields) / len(int_fields))

        humaneval_path = run_dir / "humaneval_scores.jsonl"
        if humaneval_path.exists():
            with humaneval_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    pass_at_1 = row.get("pass_at_1")
                    if pass_at_1 is not None:
                        # Rescale 0/1 → 1-5 so the column shares a scale.
                        by_cell[cell].append(1.0 + 4.0 * float(pass_at_1))
    return dict(by_cell)


# ---- Rendering ------------------------------------------------------------


def _render_headline(
    paid_by_cell: dict,
    would_have_by_cell: dict,
    latency_by_cell: dict,
    judge_by_cell: dict,
) -> str:
    """Headline table: workload × condition × model → cost, quality, latency."""
    if not paid_by_cell and not latency_by_cell:
        return (
            "| Workload | Condition | Model | $/task paid (mean ± std) | "
            "$/task saved | Quality (1-5) | p50 latency (ms) | p95 latency (ms) |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| _no data yet_ |   |   |   |   |   |   |   |\n"
        )

    lines = [
        "| Workload | Condition | Model | $/task paid (mean ± std) | "
        "$/task saved (mean ± std) | Quality (1-5, mean ± std) | "
        "p50 latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
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
        savings = (
            [w - p for w, p in zip(would_have, paid)]
            if len(paid) == len(would_have)
            else []
        )
        paid_cell = stats.fmt_mean_std(paid, precision=4, unit=" $")
        saved_cell = stats.fmt_mean_std(savings, precision=4, unit=" $")
        quality_vals = judge_by_cell.get(cell, [])
        quality_cell = stats.fmt_mean_std(quality_vals, precision=2) if quality_vals else "—"
        latencies = latency_by_cell.get(cell, [])
        p50 = f"{stats.p50(latencies):.0f}" if latencies else "—"
        p95 = f"{stats.p95(latencies):.0f}" if latencies else "—"
        lines.append(
            f"| {workload} | {condition} | {model} | {paid_cell} | {saved_cell} | "
            f"{quality_cell} | {p50} | {p95} |"
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


def _render_per_workload_deep_dive(
    paid_by_cell: dict,
    would_have_by_cell: dict,
    latency_by_cell: dict,
    counts: dict,
    judge_by_cell: dict,
) -> str:
    """Per-workload section — savings %, resolution mix, latency narrative."""
    workloads = sorted({cell[0] for cell in paid_by_cell.keys()})
    if not workloads:
        return "_filled in once workloads land._\n"

    parts: list[str] = []
    for workload in workloads:
        parts.append(f"### {workload}\n\n")

        cells_for_workload = {cell for cell in paid_by_cell if cell[0] == workload}
        conditions_present = sorted({c[1] for c in cells_for_workload})
        models_present = sorted({c[2] for c in cells_for_workload})

        # Baseline mean (averaged across models within the workload) is the
        # denominator for "reduction vs. baseline" in the cost table.
        baseline_vals: list[float] = []
        for model in models_present:
            baseline_vals.extend(paid_by_cell.get((workload, "baseline", model), []))
        avg_baseline = stats.mean(baseline_vals) if baseline_vals else 0.0

        parts.append(
            "**Cost reduction per condition** "
            "(mean across seeds and any models; warm vs. baseline tells "
            "the headline savings):\n\n"
        )
        parts.append("| Condition | Paid mean | Saved mean | Reduction vs. baseline |\n")
        parts.append("|---|---|---|---|\n")
        for condition in conditions_present:
            paid_all: list[float] = []
            saved_all: list[float] = []
            for model in models_present:
                p = paid_by_cell.get((workload, condition, model), [])
                w = would_have_by_cell.get((workload, condition, model), [])
                paid_all.extend(p)
                if len(p) == len(w):
                    saved_all.extend([wi - pi for wi, pi in zip(w, p)])
            m_paid = stats.mean(paid_all) if paid_all else 0.0
            m_saved = stats.mean(saved_all) if saved_all else 0.0
            if avg_baseline > 0:
                reduction = f"{(1 - m_paid / avg_baseline) * 100:+.1f}%"
            else:
                reduction = "—"
            parts.append(
                f"| {condition} | ${m_paid:.4f} | ${m_saved:.4f} | {reduction} |\n"
            )

        parts.append(
            "\n**Latency per condition** (p50 / p95 ms, all calls included):\n\n"
        )
        parts.append("| Condition | p50 | p95 |\n|---|---|---|\n")
        for condition in conditions_present:
            lats: list[float] = []
            for model in models_present:
                lats.extend(latency_by_cell.get((workload, condition, model), []))
            if lats:
                parts.append(
                    f"| {condition} | {stats.p50(lats):.0f} | {stats.p95(lats):.0f} |\n"
                )
            else:
                parts.append(f"| {condition} | — | — |\n")

        parts.append("\n**Quality (Opus judge, 1-5 mean ± std):**\n\n")
        parts.append("| Condition | Score |\n|---|---|\n")
        for condition in conditions_present:
            qs: list[float] = []
            for model in models_present:
                qs.extend(judge_by_cell.get((workload, condition, model), []))
            qcell = stats.fmt_mean_std(qs, precision=2) if qs else "—"
            parts.append(f"| {condition} | {qcell} |\n")
        parts.append("\n")

    return "".join(parts)


def render(records: list[dict[str, Any]], pricing: dict[str, Any]) -> str:
    """Build the full report markdown string. Pure function — no IO."""
    paid_by_cell, would_have_by_cell = _per_task_cost(records, pricing)
    latency_by_cell = _per_call_latency(records)
    counts = _resolution_counts(records)
    # Judge scores are an optional sidecar; missing = "—" in the Quality column.
    judge_by_cell = _load_judge_scores(RAW_DIR)

    pulled_on = pricing.get("pulled_on", "unknown")
    source_url = pricing.get("source_url", "")
    n_records = len(records)
    n_runs = len({r["run_id"] for r in records}) if records else 0
    workloads_with_data = sorted({r["workload"] for r in records}) if records else []
    n_workloads_total = 4  # support_triage, research_analyst, code_review, humaneval

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if workloads_with_data and len(workloads_with_data) < n_workloads_total:
        status_line = (
            f"> **Status: partial.** {len(workloads_with_data)}/{n_workloads_total} "
            f"workloads complete: {', '.join(workloads_with_data)}. Pending: "
            f"{', '.join(w for w in ['support_triage', 'research_analyst', 'code_review', 'humaneval'] if w not in workloads_with_data)}.\n>\n"
        )
    elif workloads_with_data:
        status_line = "> **Status: complete.** All planned workloads measured.\n>\n"
    else:
        status_line = "> **Status: no data.**\n>\n"

    return (
        "# ThriftAI Benchmark Results\n\n"
        + status_line
        + f"> Generated {now} from {n_records} calls across {n_runs} run(s).\n"
        f"> Pricing snapshot: pulled {pulled_on}"
        + (f" — [source]({source_url})" if source_url else "")
        + ".\n\n"
        "## Headline\n\n"
        + _render_headline(
            paid_by_cell, would_have_by_cell, latency_by_cell, judge_by_cell
        )
        + "\n## Call resolution breakdown\n\n"
        "Counts of brokered-call outcomes per cell. Cache vs replay vs live\n"
        "tells you which mechanism is doing the work.\n\n"
        + _render_resolution_breakdown(counts)
        + "\n## Per-workload deep dives\n\n"
        + _render_per_workload_deep_dive(
            paid_by_cell, would_have_by_cell, latency_by_cell, counts, judge_by_cell
        )
        + "\n"
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
