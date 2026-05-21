"""Test-suite fixtures shared across benchmarks/tests/.

The runner sets a global budget cap on instrumentation when main() is
invoked. That state leaks across tests unless we explicitly reset it,
which can cause tests that use the same in-process state (e.g.
test_support_triage's bench-context test) to read the real
persistent spend ledger and fail.

Resetting before every test keeps each test self-contained.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_instrumentation_state():
    """Reset module-level budget + context state before every test."""
    from benchmarks.runner import instrumentation

    instrumentation.configure_budget(cap_usd=None, pricing_yaml_path=None)
    instrumentation.set_context(None)
    yield
    instrumentation.configure_budget(cap_usd=None, pricing_yaml_path=None)
    instrumentation.set_context(None)
