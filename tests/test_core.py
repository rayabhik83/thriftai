"""Tests for ThriftAI core functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from thriftai.agent import (
    _agent_registry,
    agent,
    get_current_agent,
    get_dependents,
)
from thriftai.broker import Broker, CallResolution
from thriftai.cache import (
    ExactCache,
    compute_content_hash,
    compute_prompt_hash,
)
from thriftai.cost import AgentCostEntry, CostReport
from thriftai.session import Session
from thriftai.trace import Trace, TraceEntry, TraceStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_completion(content: str = "mocked response"):
    def _fn(*args, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        return resp
    return _fn


@pytest.fixture(autouse=True)
def _reset_registry():
    _agent_registry.clear()
    yield
    _agent_registry.clear()


# ---------------------------------------------------------------------------
# @agent decorator
# ---------------------------------------------------------------------------


class TestAgentDecorator:
    def test_registers_agent(self):
        @agent(name="test_agent", depends_on=["other"])
        def my_fn():
            pass

        assert "test_agent" in _agent_registry
        assert _agent_registry["test_agent"].depends_on == ["other"]

    def test_sets_thread_local(self):
        @agent(name="test_agent")
        def my_fn():
            return get_current_agent()

        result = my_fn()
        assert result == "test_agent"
        assert get_current_agent() is None

    def test_get_dependents_transitive(self):
        @agent(name="a")
        def fa():
            pass

        @agent(name="b", depends_on=["a"])
        def fb():
            pass

        @agent(name="c", depends_on=["b"])
        def fc():
            pass

        @agent(name="d", depends_on=["a"])
        def fd():
            pass

        deps = get_dependents("a")
        assert set(deps) == {"b", "c", "d"}
        assert get_dependents("c") == []


# ---------------------------------------------------------------------------
# ExactCache
# ---------------------------------------------------------------------------


class TestExactCache:
    def test_put_get_roundtrip(self, tmp_path):
        cache = ExactCache(tmp_path)
        cache.put(
            agent_name="researcher",
            prompt_hash="ph1",
            content_hash="ch1",
            model="claude",
            response_text="hello",
            input_tokens=10,
            output_tokens=5,
        )
        hit = cache.get("researcher", "ph1", "ch1")
        assert hit is not None
        assert hit["response_text"] == "hello"
        assert hit["input_tokens"] == 10

    def test_miss_returns_none(self, tmp_path):
        cache = ExactCache(tmp_path)
        assert cache.get("x", "y", "z") is None

    def test_prompt_change_invalidates(self, tmp_path):
        cache = ExactCache(tmp_path)
        cache.put("a", "p1", "c1", "m", "resp", 1, 1)
        assert cache.get("a", "p1", "c1") is not None
        assert cache.get("a", "p2", "c1") is None  # different prompt hash

    def test_invalidate_agent(self, tmp_path):
        cache = ExactCache(tmp_path)
        cache.put("a", "p", "c1", "m", "r1", 1, 1)
        cache.put("a", "p", "c2", "m", "r2", 1, 1)
        cache.put("b", "p", "c1", "m", "r3", 1, 1)
        deleted = cache.invalidate_agent("a")
        assert deleted == 2
        assert cache.get("a", "p", "c1") is None
        assert cache.get("b", "p", "c1") is not None

    def test_stats_reflects_hits(self, tmp_path):
        cache = ExactCache(tmp_path)
        cache.put("a", "p", "c", "m", "r", 1, 1)
        cache.get("a", "p", "c")
        cache.get("a", "p", "c")
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["total_hits"] == 2
        assert stats["db_size_bytes"] > 0

    def test_hash_helpers_scoped(self):
        msgs_a = [
            {"role": "system", "content": "S1"},
            {"role": "user", "content": "U"},
        ]
        msgs_b = [
            {"role": "system", "content": "S2"},
            {"role": "user", "content": "U"},
        ]
        assert compute_prompt_hash(msgs_a) != compute_prompt_hash(msgs_b)
        assert compute_content_hash(msgs_a) == compute_content_hash(msgs_b)


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------


class TestTraceStore:
    def test_record_load_roundtrip(self, tmp_path):
        store = TraceStore(tmp_path)
        trace = Trace(
            trace_id="run_test",
            entries=[
                TraceEntry(
                    sequence=0,
                    agent_name="researcher",
                    model="claude",
                    messages_hash="h",
                    response_text="out",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                )
            ],
            total_cost_usd=0.001,
        )
        store.record(trace)
        loaded = store.load("run_test")
        assert loaded.trace_id == "run_test"
        assert len(loaded.entries) == 1
        assert loaded.entries[0].response_text == "out"
        assert loaded.entries[0].input_tokens == 10

    def test_list_traces(self, tmp_path):
        store = TraceStore(tmp_path)
        store.record(Trace(trace_id="a"))
        store.record(Trace(trace_id="b"))
        assert set(store.list_traces()) == {"a", "b"}

    def test_load_missing_raises(self, tmp_path):
        store = TraceStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("does_not_exist")


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class TestBroker:
    def _make(self, tmp_path):
        cache = ExactCache(tmp_path)
        trace_store = TraceStore(tmp_path)
        return Broker(cache=cache, trace_store=trace_store, cost_tracker=None), cache

    def test_live_call_then_cache_hit(self, tmp_path):
        broker, cache = self._make(tmp_path)
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        with patch("thriftai.broker.call_litellm") as live, \
             patch("thriftai.broker.estimate_cost", return_value=0.01):
            live.return_value = MagicMock(
                response_text="R",
                model="m",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.01,
            )
            first = broker.route(messages, "m", agent_name="a1")
            assert first.resolution is CallResolution.LIVE
            assert live.call_count == 1

            second = broker.route(messages, "m", agent_name="a1")
            assert second.resolution is CallResolution.CACHE_HIT
            assert second.response_text == "R"
            # No additional live call
            assert live.call_count == 1

    def test_replay_serves_from_trace(self, tmp_path):
        broker, _cache = self._make(tmp_path)
        trace = Trace(
            trace_id="t",
            entries=[
                TraceEntry(
                    sequence=0,
                    agent_name="a1",
                    model="m",
                    messages_hash="h",
                    response_text="traced",
                    input_tokens=1,
                    output_tokens=1,
                )
            ],
        )
        messages = [{"role": "user", "content": "anything"}]
        with patch("thriftai.broker.call_litellm") as live, \
             patch("thriftai.broker.estimate_cost", return_value=0.02):
            result = broker.route(
                messages, "m", agent_name="a1",
                replay_trace=trace, live_agents=[],
            )
        assert result.resolution is CallResolution.REPLAY
        assert result.response_text == "traced"
        assert result.cost_usd == 0.0
        assert result.cached_cost_usd == 0.02
        live.assert_not_called()

    def test_replay_live_agent_goes_through(self, tmp_path):
        broker, _cache = self._make(tmp_path)
        trace = Trace(
            trace_id="t",
            entries=[
                TraceEntry(
                    sequence=0, agent_name="a1", model="m",
                    messages_hash="h", response_text="traced",
                    input_tokens=1, output_tokens=1,
                )
            ],
        )
        messages = [{"role": "user", "content": "U"}]
        with patch("thriftai.broker.call_litellm") as live:
            live.return_value = MagicMock(
                response_text="fresh", model="m",
                input_tokens=1, output_tokens=1, cost_usd=0.01,
            )
            result = broker.route(
                messages, "m", agent_name="a1",
                replay_trace=trace, live_agents=["a1"],
            )
        assert result.resolution is CallResolution.LIVE
        assert result.response_text == "fresh"


# ---------------------------------------------------------------------------
# Session / end-to-end
# ---------------------------------------------------------------------------


class TestSession:
    def _mk_agents(self):
        @agent(name="researcher")
        def research(session, topic):
            return session.completion(
                messages=[
                    {"role": "system", "content": "You research."},
                    {"role": "user", "content": topic},
                ],
                model="anthropic/claude",
            )

        @agent(name="writer", depends_on=["researcher"])
        def write(session, data):
            return session.completion(
                messages=[
                    {"role": "system", "content": "You write."},
                    {"role": "user", "content": data},
                ],
                model="anthropic/claude",
            )

        return research, write

    def test_run_records_trace_and_costs(self, tmp_path):
        research, write = self._mk_agents()
        session = Session(cache_dir=tmp_path)

        with patch("thriftai.broker.call_litellm",
                   side_effect=[
                       MagicMock(response_text="R1", model="m",
                                 input_tokens=1, output_tokens=1, cost_usd=0.05),
                       MagicMock(response_text="R2", model="m",
                                 input_tokens=1, output_tokens=1, cost_usd=0.07),
                   ]):
            with session.run() as run:
                data = research(run, "AI costs")
                summary = write(run, data)
                trace_id = run.trace_id

        assert data == "R1"
        assert summary == "R2"

        loaded = session.trace_store.load(trace_id)
        assert [e.agent_name for e in loaded.entries] == ["researcher", "writer"]

    def test_replay_serves_traced_agent_and_live_only_runs_writer(self, tmp_path):
        research, write = self._mk_agents()
        session = Session(cache_dir=tmp_path)

        # First run: both go live
        with patch("thriftai.broker.call_litellm",
                   side_effect=[
                       MagicMock(response_text="R1", model="m",
                                 input_tokens=1, output_tokens=1, cost_usd=0.05),
                       MagicMock(response_text="R2", model="m",
                                 input_tokens=1, output_tokens=1, cost_usd=0.07),
                   ]):
            with session.run() as run:
                research(run, "topic")
                write(run, "d")
                trace_id = run.trace_id

        # Replay: researcher replays, writer goes live
        with patch("thriftai.broker.call_litellm") as live, \
             patch("thriftai.broker.estimate_cost", return_value=0.05):
            live.return_value = MagicMock(
                response_text="R2-NEW", model="m",
                input_tokens=1, output_tokens=1, cost_usd=0.07,
            )
            with session.replay(trace_id=trace_id, live=["writer"]) as run:
                r_out = research(run, "topic")
                w_out = write(run, r_out)

            # Writer ran live; researcher replayed
            assert live.call_count == 1
            assert r_out == "R1"
            assert w_out == "R2-NEW"

            # Cost report: researcher=replay, writer=live
            resolutions = [e.resolution for e in run.cost_report.entries]
            assert resolutions == ["replay", "live"]
            assert run.cost_report.total_saved > 0

    def test_replay_invalidates_downstream_on_divergence(self, tmp_path):
        research, write = self._mk_agents()
        session = Session(cache_dir=tmp_path)

        with patch("thriftai.broker.call_litellm",
                   side_effect=[
                       MagicMock(response_text="R1-OLD", model="m",
                                 input_tokens=1, output_tokens=1, cost_usd=0.05),
                       MagicMock(response_text="R2-OLD", model="m",
                                 input_tokens=1, output_tokens=1, cost_usd=0.07),
                   ]):
            with session.run() as run:
                research(run, "topic")
                write(run, "d")
                trace_id = run.trace_id

        # Simulate the developer changing researcher's prompt: wipe its cache
        # so the replay forces a live call (which will diverge from the trace).
        session.cache.invalidate_agent("researcher")
        session.cache.invalidate_agent("writer")

        # Replay with researcher live producing DIFFERENT output.
        # writer was not marked live, but should be invalidated
        # because researcher diverged — so writer goes through cache → live.
        with patch("thriftai.broker.call_litellm") as live:
            live.side_effect = [
                MagicMock(response_text="R1-NEW", model="m",
                          input_tokens=1, output_tokens=1, cost_usd=0.05),
                MagicMock(response_text="R2-NEW", model="m",
                          input_tokens=1, output_tokens=1, cost_usd=0.07),
            ]
            with session.replay(trace_id=trace_id, live=["researcher"]) as run:
                r_out = research(run, "topic")
                w_out = write(run, r_out)

            assert r_out == "R1-NEW"
            # Writer invalidated -> went live (trace output would have been R2-OLD)
            assert w_out == "R2-NEW"
            assert live.call_count == 2
            assert "writer" in run.invalidated_agents


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------


class TestCostReport:
    def test_summary_shows_savings(self):
        report = CostReport(
            entries=[
                AgentCostEntry(
                    agent_name="researcher",
                    resolution="replay",
                    model="m",
                    actual_cost_usd=0.0,
                    would_have_cost_usd=0.36,
                ),
                AgentCostEntry(
                    agent_name="writer",
                    resolution="live",
                    model="m",
                    actual_cost_usd=0.07,
                    would_have_cost_usd=0.07,
                ),
            ]
        )
        text = report.summary()
        assert "researcher" in text
        assert "writer" in text
        assert "replay" in text
        assert "live" in text
        assert "0.36" in text
        assert report.total_cost == pytest.approx(0.07)
        assert report.total_saved == pytest.approx(0.36)
