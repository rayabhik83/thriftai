"""
Thin statistical utilities for the benchmark report.

We deliberately keep this small: mean, std (ddof=1), p50/p95, and a
bootstrap CI. The point is to use these everywhere in the report
rather than scattering ad-hoc numpy calls; if the methodology section
says "mean ± std (N=5)", every cell in the table goes through these
functions.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import bootstrap


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def std(values: Iterable[float], ddof: int = 1) -> float:
    """Sample std (ddof=1). Returns 0.0 for a single-value input."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=ddof))


def percentile(values: Iterable[float], q: float) -> float:
    """q in [0, 100]."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.percentile(arr, q))


def p50(values: Iterable[float]) -> float:
    return percentile(values, 50)


def p95(values: Iterable[float]) -> float:
    return percentile(values, 95)


def bootstrap_ci(
    values: Sequence[float],
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the mean. Returns (lo, hi).

    Falls back to (nan, nan) for empty input and (mean, mean) for
    single-value input — both genuinely have undefined CIs but we
    want the report to render without crashing.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (arr,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=rng,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def fmt_mean_std(values: Iterable[float], precision: int = 4, unit: str = "") -> str:
    """Render a 'mean ± std' string for table cells. Empty → '—'."""
    vals = list(values)
    if not vals:
        return "—"
    m = mean(vals)
    s = std(vals)
    fmt = f"{{:.{precision}f}}"
    body = f"{fmt.format(m)} ± {fmt.format(s)}"
    return f"{body}{unit}" if unit else body
