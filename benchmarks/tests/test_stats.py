"""Smoke tests for the stats utilities."""

from __future__ import annotations

import math

import pytest

from benchmarks.runner import stats


def test_mean_and_std():
    assert stats.mean([1.0, 2.0, 3.0]) == 2.0
    # Sample std (ddof=1) of [1, 2, 3] is 1.0.
    assert stats.std([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_empty_inputs_dont_crash():
    assert math.isnan(stats.mean([]))
    assert stats.std([]) == 0.0
    assert math.isnan(stats.percentile([], 50))


def test_single_value_inputs():
    """std of a single value is 0; p50/p95 collapse to the value."""
    assert stats.std([42.0]) == 0.0
    assert stats.p50([42.0]) == 42.0
    assert stats.p95([42.0]) == 42.0


def test_bootstrap_ci_brackets_the_mean():
    """The CI for a uniform sample should bracket the sample mean."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = stats.bootstrap_ci(values, n_resamples=1000, seed=42)
    m = sum(values) / len(values)
    assert lo <= m <= hi


def test_bootstrap_ci_handles_degenerate_inputs():
    lo, hi = stats.bootstrap_ci([], n_resamples=10)
    assert math.isnan(lo) and math.isnan(hi)
    # Single value → both bounds equal that value.
    lo, hi = stats.bootstrap_ci([7.0], n_resamples=10)
    assert lo == 7.0 and hi == 7.0


def test_fmt_mean_std_empty():
    assert stats.fmt_mean_std([]) == "—"


def test_fmt_mean_std_with_unit():
    out = stats.fmt_mean_std([0.001, 0.002, 0.003], precision=4, unit=" $")
    assert "$" in out
    assert "±" in out
