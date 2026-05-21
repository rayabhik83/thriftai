"""
Benchmark runner — orchestrates conditions × seeds × tasks for one workload.

Run from the repo root:

    python -m benchmarks.runner.run --workload support_triage --n 2

What it does:

1. Loads the workload config (`benchmarks/configs/<workload>.yaml`).
2. Loads the tasks dataset referenced in the config.
3. Installs the instrumentation patches.
4. For each (condition, seed, task) cell:
   - Build a fresh Session for the condition.
   - For `thriftai_warm`, do an unmeasured warmup pass over all tasks
     to populate the cache.
   - Set the bench context.
   - Run the workload's `run_one(session, task, ...)` and collect
     the produced artifacts.
5. Writes per-call JSONL to `benchmarks/results/raw/<run_id>/calls.jsonl`.
6. After all cells complete, regenerates `REPORT.md`.

API-spend safety: this script makes real LLM calls and costs real money.
It's invoked manually (Makefile targets `smoke` and `bench`), never as a
side effect of running tests.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import instrumentation
from .conditions import make_session, reset_cache_dir
from .report import main as render_report

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
CACHE_ROOT = BENCH_DIR / "cache" / "sessions"


# ---- workload registry ----------------------------------------------------


def import_workload(workload: str):
    return importlib.import_module(f"benchmarks.workloads.{workload}")


def load_workload_config(workload: str) -> dict[str, Any]:
    config_path = BENCH_DIR / "configs" / f"{workload}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Expected one of {list((BENCH_DIR / 'configs').glob('*.yaml'))}."
        )
    with config_path.open() as f:
        return yaml.safe_load(f)


# ---- run-id and JSONL helpers ---------------------------------------------


def _make_run_id(workload: str, condition: str, model: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{workload}_{condition}_{model}_seed{seed}"


def _jsonl_path_for(run_id: str) -> Path:
    return RAW_DIR / run_id / "calls.jsonl"


# ---- cell execution -------------------------------------------------------


def _run_one_cell(
    workload_mod,
    workload_config: dict[str, Any],
    condition: str,
    model: str,
    seed: int,
    tasks: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run all tasks for a single (workload, condition, model, seed) cell.

    Returns a dict with the per-task artifacts (judging happens elsewhere).
    """
    run_id = _make_run_id(workload_config["workload"], condition, model, seed)
    cache_dir = CACHE_ROOT / workload_config["workload"] / condition / f"seed_{seed}"

    print(f"  cell: {condition} seed={seed} model={model} ({len(tasks)} tasks)")
    sys.stdout.flush()

    # Cold start guarantee — each measured run begins with a fresh cache
    # except when the warm pre-population step runs first (handled below).
    reset_cache_dir(cache_dir)

    # If this is the warm condition, do an unmeasured warmup pass before
    # the measured pass. The warmup pass uses the same cache_dir; the
    # measured pass then runs against an already-warm cache.
    if condition == "thriftai_warm":
        warmup_session = make_session(
            "thriftai_cold",  # warmup uses the same factory as cold
            cache_dir,
            workload_config.get("thriftai_session", {}),
        )
        instrumentation.set_context(None)  # un-measured pass — no JSONL writes
        for task in tasks:
            workload_mod.run_one(warmup_session, task, corpus, model)

    # Now build the session we measure.
    session = make_session(condition, cache_dir, workload_config.get("thriftai_session", {}))

    # Configure JSONL output for this cell and set the bench context for
    # each per-task invocation.
    jsonl_path = _jsonl_path_for(run_id)
    instrumentation.configure_output(jsonl_path)

    artifacts: list[dict[str, Any]] = []
    for task in tasks:
        instrumentation.set_context(
            instrumentation.BenchContext(
                run_id=run_id,
                workload=workload_config["workload"],
                condition=condition,
                model_under_test=model,
                task_id=task["id"],
                seed=seed,
            )
        )
        try:
            result = workload_mod.run_one(session, task, corpus, model)
        finally:
            instrumentation.set_context(None)
        artifacts.append(result)

    return {
        "run_id": run_id,
        "condition": condition,
        "model": model,
        "seed": seed,
        "artifacts": artifacts,
    }


# ---- entry point ----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="benchmarks.runner.run")
    p.add_argument("--workload", required=True, help="workload name or 'all'")
    p.add_argument("--n", type=int, default=5, help="number of seeds per cell")
    p.add_argument(
        "--model",
        default=None,
        help="override the default model from the workload config",
    )
    p.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="for smoke: only process the first N tasks",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=10.0,
        help=(
            "Hard cap on cumulative spend (USD) across all runs in this branch. "
            "Persistent across invocations via benchmarks/cache/spend_ledger.jsonl. "
            "Default: 10."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Auto-load benchmarks/.env if present so ANTHROPIC_API_KEY is available
    # without the user needing to source it manually.
    load_dotenv(BENCH_DIR / ".env")

    # Enable LiteLLM's built-in 429 retry with exponential backoff. The
    # benchmark hits the per-minute rate limit on burst, especially in
    # baseline / thriftai_cold conditions where every call is live.
    # Retried calls' latency naturally inflates — that's accurate, since
    # a real user hitting the same rate limit experiences the same wait.
    import litellm
    litellm.num_retries = 5

    instrumentation.install()

    # Spend tracking + cap. Total accrued lives in benchmarks/cache/
    # spend_ledger.jsonl (gitignored). Persists across runs so the
    # $10 budget is a project-wide cap, not per-invocation.
    from . import budget as _budget
    instrumentation.configure_budget(
        cap_usd=args.budget,
        pricing_yaml_path=BENCH_DIR / "pricing.yaml",
    )
    before = _budget.total_spent()
    print(f"budget: spent so far ${before:.4f}, cap ${args.budget:.2f}")

    workloads = (
        ["support_triage", "research_analyst", "code_review", "humaneval"]
        if args.workload == "all"
        else [args.workload]
    )

    for workload in workloads:
        print(f"workload: {workload}")
        config = load_workload_config(workload)
        workload_mod = import_workload(workload)

        data_file = REPO_ROOT / config["data_file"]
        if not data_file.exists():
            print(
                f"  SKIP: data file missing: {data_file}\n"
                f"        Run the data synthesis step for this workload first.",
                file=sys.stderr,
            )
            continue

        corpus = workload_mod.load_tasks(data_file)
        tasks = corpus if args.task_limit is None else corpus[: args.task_limit]

        model = args.model or config["default_model"]

        for seed in range(args.n):
            for condition in config["conditions"]:
                _run_one_cell(
                    workload_mod=workload_mod,
                    workload_config=config,
                    condition=condition,
                    model=model,
                    seed=seed,
                    tasks=tasks,
                    corpus=corpus,
                )

    # Regenerate the report from whatever's now in raw/.
    render_report()

    after = _budget.total_spent()
    print(f"budget: spent this run ${after - before:.4f}, total ${after:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
