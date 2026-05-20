"""
Verify the instrumentation harness against a mocked LiteLLM.

Mocks `litellm.completion` (not anywhere inside ThriftAI) so the
full chain runs:

    benchmarks instrumented Broker.route
        → original Broker.route
            → benchmarks instrumented call_litellm
                → original thriftai.providers.call_litellm
                    → litellm.completion (MOCKED)

This means a passing test verifies the patch installation, the JSONL
writer, the context propagation, AND that ThriftAI's broker actually
calls into our patched functions correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.runner import instrumentation
from thriftai import Session, agent


def _make_fake_litellm_response(content: str, prompt_tokens: int, completion_tokens: int):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return response


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "calls.jsonl"
    instrumentation.configure_output(p)
    return p


@pytest.fixture(autouse=True)
def installed_patches():
    instrumentation.install()
    yield
    instrumentation.uninstall()
    instrumentation.set_context(None)


@pytest.fixture
def bench_ctx():
    ctx = instrumentation.BenchContext(
        run_id="test_run",
        workload="test_workload",
        condition="baseline",
        model_under_test="claude-haiku-4-5",
        task_id="task_0",
        seed=42,
    )
    instrumentation.set_context(ctx)
    yield ctx
    instrumentation.set_context(None)


def test_install_uninstall_is_idempotent():
    # Double install is a no-op (already installed via fixture).
    instrumentation.install()
    instrumentation.install()
    assert instrumentation.is_installed()
    instrumentation.uninstall()
    instrumentation.uninstall()
    assert not instrumentation.is_installed()
    # Restore for the rest of the suite.
    instrumentation.install()


def test_no_records_outside_bench_context(tmp_jsonl: Path, tmp_path: Path):
    """Without set_context(), instrumentation is a clean passthrough."""

    @agent(name="greeter1")
    def greet(run, name):
        return run.completion(
            messages=[{"role": "user", "content": f"hello {name}"}],
            model="claude-haiku-4-5",
        )

    session = Session(cache_dir=tmp_path / "ta_cache_a", enabled=False)
    fake = _make_fake_litellm_response("Hi!", 4, 2)

    with patch("litellm.completion", return_value=fake), patch(
        "litellm.completion_cost", return_value=0.0
    ):
        with session.run() as run:
            out = greet(run, "world")

    assert out == "Hi!"
    assert not tmp_jsonl.exists() or tmp_jsonl.read_text() == ""


def test_records_a_live_call(tmp_jsonl: Path, tmp_path: Path, bench_ctx):
    """One live completion produces one JSONL line with expected fields."""

    @agent(name="greeter2")
    def greet(run, name):
        return run.completion(
            messages=[{"role": "user", "content": f"hello {name}"}],
            model="claude-haiku-4-5",
        )

    session = Session(cache_dir=tmp_path / "ta_cache_b", enabled=False)
    fake = _make_fake_litellm_response("Hello from fake LLM", 42, 8)

    with patch("litellm.completion", return_value=fake), patch(
        "litellm.completion_cost", return_value=0.000123
    ):
        with session.run() as run:
            out = greet(run, "world")

    assert out == "Hello from fake LLM"

    lines = tmp_jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # Core context fields.
    assert record["run_id"] == "test_run"
    assert record["workload"] == "test_workload"
    assert record["condition"] == "baseline"
    assert record["task_id"] == "task_0"
    assert record["seed"] == 42

    # ThriftAI fields.
    assert record["agent_name"] == "greeter2"
    assert record["broker_resolution"] == "live"
    assert record["input_tokens"] == 42
    assert record["output_tokens"] == 8

    # Latency: total > 0; api was actually called so api_ms is set;
    # overhead is non-negative.
    assert record["latency_total_ms"] >= 0
    assert record["latency_api_ms"] is not None and record["latency_api_ms"] >= 0
    assert record["latency_overhead_ms"] >= 0

    # Hash format.
    assert record["response_text_hash"].startswith("sha256:")
    assert len(record["response_text_hash"]) == len("sha256:") + 16


def test_records_a_cache_hit(tmp_jsonl: Path, tmp_path: Path, bench_ctx):
    """Second identical call returns a cache_hit record with no API latency."""

    @agent(name="echoer")
    def echo(run, x):
        return run.completion(
            messages=[{"role": "user", "content": x}],
            model="claude-haiku-4-5",
        )

    # enabled=True so the exact cache is wired up (not _NoOpCache).
    session = Session(cache_dir=tmp_path / "ta_cache_c", enabled=True)
    fake = _make_fake_litellm_response("echo", 10, 5)

    with patch("litellm.completion", return_value=fake), patch(
        "litellm.completion_cost", return_value=0.0001
    ):
        with session.run() as run:
            echo(run, "same input")
        with session.run() as run:
            echo(run, "same input")

    lines = tmp_jsonl.read_text().strip().splitlines()
    assert len(lines) == 2
    first, second = json.loads(lines[0]), json.loads(lines[1])

    assert first["broker_resolution"] == "live"
    assert first["latency_api_ms"] is not None

    assert second["broker_resolution"] == "cache_hit"
    # Cache hit doesn't make an API call, so api latency was never set.
    assert second["latency_api_ms"] is None
    assert second["input_tokens"] == 10  # came from cache entry
    assert second["output_tokens"] == 5
