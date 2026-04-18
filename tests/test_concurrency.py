"""
Concurrency tests.

What we're protecting against:

1. **Cross-thread agent isolation.** `_current_agent` is a `threading.local`.
   Two agents on two threads must not see each other's state.

2. **Nested @agent calls.** Calling agent B from inside agent A must restore
   `get_current_agent() == "A"` when B returns. Today the decorator sets the
   thread-local to `None` in its `finally`, which silently clobbers the
   outer agent. This test asserts the *correct* behavior; if it fails we fix
   the decorator rather than the test.

3. **Concurrent broker LIVE calls.** `ExactCache` opens its SQLite connection
   with `check_same_thread=False` and shares it across threads. Parallel
   `put()` calls must not corrupt or lose entries.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from thriftai.agent import _agent_registry, agent, get_current_agent
from thriftai.broker import Broker, CallResolution
from thriftai.cache import ExactCache
from thriftai.trace import TraceStore


@pytest.fixture(autouse=True)
def _reset_registry():
    _agent_registry.clear()
    yield
    _agent_registry.clear()


# ---------------------------------------------------------------------------
# 1. Cross-thread isolation
# ---------------------------------------------------------------------------


def test_cross_thread_agent_isolation():
    """Two agents on two threads each observe their own _current_agent."""
    observed: dict[str, str | None] = {}
    barrier = threading.Barrier(2)

    @agent(name="alpha")
    def alpha_fn() -> None:
        # Both threads arrive here, then race to read the thread-local.
        barrier.wait(timeout=5)
        time.sleep(0.01)  # give the scheduler a chance to interleave
        observed["alpha"] = get_current_agent()

    @agent(name="beta")
    def beta_fn() -> None:
        barrier.wait(timeout=5)
        time.sleep(0.01)
        observed["beta"] = get_current_agent()

    t1 = threading.Thread(target=alpha_fn)
    t2 = threading.Thread(target=beta_fn)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert observed["alpha"] == "alpha"
    assert observed["beta"] == "beta"


# ---------------------------------------------------------------------------
# 2. Nested @agent calls
# ---------------------------------------------------------------------------


def test_nested_agents_restore_outer_agent_on_return():
    """When B returns to A, get_current_agent() must still be 'A'."""
    captured: dict[str, str | None] = {}

    @agent(name="inner")
    def inner_fn() -> None:
        captured["inner"] = get_current_agent()

    @agent(name="outer")
    def outer_fn() -> None:
        captured["before_inner"] = get_current_agent()
        inner_fn()
        captured["after_inner"] = get_current_agent()

    outer_fn()

    assert captured["before_inner"] == "outer"
    assert captured["inner"] == "inner"
    assert captured["after_inner"] == "outer", (
        "inner agent leaked its exit state — @agent doesn't save/restore "
        "the previous _current_agent value"
    )
    # After the outermost agent returns, the slot should be clear again.
    assert get_current_agent() is None


def test_nested_three_levels_deep():
    """Even 3-deep nesting must unwind correctly."""
    captured: list[str | None] = []

    @agent(name="c")
    def c_fn() -> None:
        captured.append(get_current_agent())

    @agent(name="b")
    def b_fn() -> None:
        captured.append(get_current_agent())
        c_fn()
        captured.append(get_current_agent())

    @agent(name="a")
    def a_fn() -> None:
        captured.append(get_current_agent())
        b_fn()
        captured.append(get_current_agent())

    a_fn()
    assert captured == ["a", "b", "c", "b", "a"]
    assert get_current_agent() is None


# ---------------------------------------------------------------------------
# 3. Concurrent broker LIVE calls
# ---------------------------------------------------------------------------


def test_concurrent_live_calls_populate_cache_without_loss(tmp_path):
    """10 threads × 10 distinct inputs each → 100 unique cache entries."""
    cache = ExactCache(tmp_path)
    broker = Broker(
        cache=cache,
        trace_store=TraceStore(tmp_path),
        cost_tracker=None,
    )

    def fake_completion(messages, model, **kwargs):
        # Return a unique response per message content so we can verify
        # the right payload lands in the right cache entry.
        content = messages[-1]["content"]
        result = MagicMock()
        result.response_text = f"reply:{content}"
        result.model = model
        result.input_tokens = 1
        result.output_tokens = 1
        result.cost_usd = 0.001
        return result

    def worker(thread_id: int) -> list[str]:
        outputs: list[str] = []
        for i in range(10):
            messages = [
                {"role": "system", "content": "S"},
                {"role": "user", "content": f"t{thread_id}-i{i}"},
            ]
            result = broker.route(
                messages=messages, model="m", agent_name=f"agent-{thread_id}"
            )
            assert result.resolution is CallResolution.LIVE
            outputs.append(result.response_text)
        return outputs

    # Patch once at test scope: mock.patch itself mutates module globals and
    # is not thread-safe when entered concurrently from worker threads.
    with patch("thriftai.broker.call_litellm", side_effect=fake_completion):
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker, tid) for tid in range(10)]
            all_outputs: list[str] = []
            for f in as_completed(futures):
                all_outputs.extend(f.result())

    assert len(all_outputs) == 100
    # Every response we got back should correspond to its own input
    assert sorted(all_outputs) == sorted(
        f"reply:t{t}-i{i}" for t in range(10) for i in range(10)
    )
    # Cache should now have exactly 100 entries (one per (agent, content))
    assert cache.stats()["total_entries"] == 100


def test_concurrent_reads_on_warm_cache(tmp_path):
    """After the cache is warm, many threads reading the same key should
    all hit and the hit_count should equal the number of reads."""
    cache = ExactCache(tmp_path)
    cache.put("a", "p", "c", "m", "R", 1, 1)

    reads = 50

    def reader() -> dict | None:
        return cache.get("a", "p", "c")

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: reader(), range(reads)))

    assert all(r is not None and r["response_text"] == "R" for r in results)
    # Every get() increments hit_count; under concurrent writes to the same
    # row SQLite serializes them, so we expect no lost updates.
    assert cache.stats()["total_hits"] == reads
