"""
Instrumentation harness for the benchmark suite.

Monkey-patches `thriftai.broker.Broker.route` and `thriftai.broker.call_litellm`
to record every brokered call into a JSONL file. The patches are idempotent
and no-op outside of an active `BenchContext`, so they're safe to leave
installed for the lifetime of a process.

Why monkey-patch and not edit the library:

- The benchmark suite must not change ThriftAI's runtime behavior. Any
  behavior change in the library would invalidate the comparison.
- Users of the library never see this code; it lives entirely under
  `benchmarks/`.

The single source of truth for "what happened" is the JSONL output. The
final REPORT.md is computed from these files plus `pricing.yaml`.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import thriftai.broker as broker_mod
import yaml

# Imported lazily inside functions to avoid a circular import at module
# load time (budget.py is small, no actual cycle, but staying conservative).


@dataclass
class BenchContext:
    """Per-task metadata that the runner attaches to every recorded call."""

    run_id: str
    workload: str
    condition: str
    model_under_test: str
    task_id: str
    seed: int


# Context lives on a ContextVar so it's safe across threads / async if we
# ever go there. When unset, instrumentation passes through without writing.
_bench_ctx: contextvars.ContextVar[BenchContext | None] = contextvars.ContextVar(
    "thriftai_bench_ctx", default=None
)

# API latency for the most recent call_litellm in this context.
_api_latency_ms: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "thriftai_bench_api_latency_ms", default=None
)

_jsonl_lock = threading.Lock()
_jsonl_path: Path | None = None

# Live-call throttle. LiteLLM's retry-with-backoff isn't enough on a tight
# per-minute rate limit because the retries keep accumulating into the same
# 60-second window. A small enforced gap *between* live calls keeps the
# average rate safely under the cap. The throttle sleeps AFTER the call,
# so it doesn't inflate recorded latency_total_ms.
_min_gap_between_live_calls_sec: float = 0.0


def configure_throttle(min_gap_sec: float) -> None:
    """Set the minimum delay enforced after every live brokered call."""
    global _min_gap_between_live_calls_sec
    _min_gap_between_live_calls_sec = float(min_gap_sec)


# Budget enforcement. Set via configure_budget(); checked after every live call.
_budget_cap_usd: float | None = None
_pricing_models: dict | None = None  # parsed pricing.yaml → models dict


def configure_budget(cap_usd: float | None, pricing_yaml_path: Path | None) -> None:
    """Enable per-call spend tracking + cap enforcement.

    cap_usd of None disables enforcement (live calls still accrue to the
    ledger so a future cap can be applied retroactively).
    """
    global _budget_cap_usd, _pricing_models
    _budget_cap_usd = cap_usd
    if pricing_yaml_path is not None and pricing_yaml_path.exists():
        with pricing_yaml_path.open() as f:
            _pricing_models = (yaml.safe_load(f) or {}).get("models", {})
    else:
        _pricing_models = None


def _cost_from_pricing(model: str, input_tokens: int, output_tokens: int) -> float:
    if _pricing_models is None:
        return 0.0
    entry = _pricing_models.get(model)
    if entry is None:
        return 0.0
    return (
        input_tokens / 1_000_000.0 * entry["input_per_million_usd"]
        + output_tokens / 1_000_000.0 * entry["output_per_million_usd"]
    )


def set_context(ctx: BenchContext | None) -> None:
    """Set or clear the active bench context for this thread/task."""
    _bench_ctx.set(ctx)


def get_context() -> BenchContext | None:
    return _bench_ctx.get()


def configure_output(jsonl_path: str | Path) -> None:
    """Set the destination JSONL path. Creates parent dirs if needed."""
    global _jsonl_path
    _jsonl_path = Path(jsonl_path)
    _jsonl_path.parent.mkdir(parents=True, exist_ok=True)


def get_output_path() -> Path | None:
    return _jsonl_path


# ---- patch state ----------------------------------------------------------

_original_route: Callable[..., Any] | None = None
_original_call_litellm: Callable[..., Any] | None = None
_installed = False


def install() -> None:
    """Idempotently install the instrumentation patches."""
    global _original_route, _original_call_litellm, _installed
    if _installed:
        return

    _original_route = broker_mod.Broker.route
    _original_call_litellm = broker_mod.call_litellm

    broker_mod.Broker.route = _instrumented_route  # type: ignore[assignment]
    broker_mod.call_litellm = _instrumented_call_litellm
    _installed = True


def uninstall() -> None:
    """Restore the original functions. Primarily used in tests."""
    global _installed
    if not _installed:
        return
    if _original_route is not None:
        broker_mod.Broker.route = _original_route  # type: ignore[assignment]
    if _original_call_litellm is not None:
        broker_mod.call_litellm = _original_call_litellm
    _installed = False


def is_installed() -> bool:
    return _installed


# ---- patched functions ----------------------------------------------------


def _instrumented_call_litellm(*args: Any, **kwargs: Any) -> Any:
    """Wrap the original call_litellm to capture API latency."""
    assert _original_call_litellm is not None
    t0 = _time.perf_counter()
    try:
        return _original_call_litellm(*args, **kwargs)
    finally:
        _api_latency_ms.set((_time.perf_counter() - t0) * 1000.0)


def _instrumented_route(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Wrap Broker.route to time the cascade and write a JSONL record."""
    assert _original_route is not None
    ctx = _bench_ctx.get()
    if ctx is None:
        # No active bench context — pass through cleanly so this patch
        # is safe to leave installed during library use.
        return _original_route(self, *args, **kwargs)

    # Reset so api latency below is fresh for this call.
    _api_latency_ms.set(None)
    t0 = _time.perf_counter()
    result = _original_route(self, *args, **kwargs)
    total_ms = (_time.perf_counter() - t0) * 1000.0
    api_ms = _api_latency_ms.get()
    overhead_ms = total_ms - (api_ms or 0.0)

    agent_name = kwargs.get("agent_name")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "workload": ctx.workload,
        "condition": ctx.condition,
        "model_under_test": ctx.model_under_test,
        "task_id": ctx.task_id,
        "seed": ctx.seed,
        "agent_name": agent_name,
        "model": result.model,
        "broker_resolution": result.resolution.value,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        # litellm-reported cost — recorded for sanity checking against
        # the canonical pricing.yaml-derived numbers in the report.
        "actual_cost_usd_litellm": result.cost_usd,
        "would_have_cost_usd_litellm": result.cached_cost_usd,
        "embedding_cost_usd_litellm": result.embedding_cost_usd,
        "similarity_score": result.similarity_score,
        "latency_total_ms": round(total_ms, 3),
        "latency_api_ms": round(api_ms, 3) if api_ms is not None else None,
        "latency_overhead_ms": round(overhead_ms, 3),
        # 16-hex-char prefix of SHA-256 is enough to detect non-determinism
        # without dumping potentially-sensitive response bodies into logs.
        "response_text_hash": "sha256:"
        + hashlib.sha256(result.response_text.encode("utf-8")).hexdigest()[:16],
    }
    _write_record(record)

    # Spend tracking — only live calls cost real money. Append to the
    # persistent ledger and abort the run cleanly if the cap is hit.
    if result.resolution.value == "live":
        from . import budget as _budget  # local import to keep startup light
        amount = _cost_from_pricing(result.model, result.input_tokens, result.output_tokens)
        _budget.record(amount, run_id=ctx.run_id, condition=ctx.condition)
        if _budget_cap_usd is not None:
            _budget.check(_budget_cap_usd)

        # Throttle the *next* live call. The sleep is AFTER the recorded
        # latency timing above, so it doesn't inflate latency_total_ms.
        if _min_gap_between_live_calls_sec > 0:
            _time.sleep(_min_gap_between_live_calls_sec)

    return result


def _write_record(record: dict) -> None:
    """Append one JSONL line to the configured output path."""
    if _jsonl_path is None:
        return
    with _jsonl_lock:
        with _jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# Convenience for tests / debugging.
def context_as_dict(ctx: BenchContext) -> dict:
    return asdict(ctx)
