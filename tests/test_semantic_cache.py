"""Tests for the semantic cache and its integration with the broker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thriftai.agent import _agent_registry, agent
from thriftai.broker import Broker, CallResolution
from thriftai.cache import ExactCache, compute_content_hash, compute_prompt_hash
from thriftai.cache.semantic import (
    SemanticCache,
    _cosine_similarity,
    _messages_to_text,
)
from thriftai.session import Session
from thriftai.trace import TraceStore


# ---------------------------------------------------------------------------
# Embedding mocking utilities
# ---------------------------------------------------------------------------


def _make_embedding_response(vector: np.ndarray, cost: float = 0.00001) -> MagicMock:
    """Build a MagicMock shaped like a litellm.embedding() response."""
    resp = MagicMock()
    resp.data = [{"embedding": vector.tolist()}]
    # `litellm.completion_cost(completion_response=resp)` is used for cost.
    return resp


class _EmbeddingScripter:
    """Scripts deterministic embeddings per input text.

    Any text not registered gets a unique random vector (so we don't
    accidentally collide and produce misleading similarity results).
    """

    def __init__(self, dim: int = 8, cost_per_call: float = 0.00001):
        self.dim = dim
        self.cost_per_call = cost_per_call
        self.by_text: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(42)

    def register(self, text: str, vector: np.ndarray) -> None:
        self.by_text[text] = vector.astype(np.float32)

    def __call__(self, *args, **kwargs):
        inputs = kwargs.get("input") or args[1]
        text = inputs[0]
        vec = self.by_text.get(text)
        if vec is None:
            vec = self._rng.standard_normal(self.dim).astype(np.float32)
        return _make_embedding_response(vec)


@pytest.fixture(autouse=True)
def _reset_registry():
    _agent_registry.clear()
    yield
    _agent_registry.clear()


# ---------------------------------------------------------------------------
# Unit tests for SemanticCache
# ---------------------------------------------------------------------------


class TestSemanticCache:
    def _cache(self, tmp_path, threshold: float = 0.92) -> SemanticCache:
        return SemanticCache(
            db_path=tmp_path / "cache.db",
            embedding_model="text-embedding-3-small",
            threshold=threshold,
            # These tests use short queries; opt out of the cost-control
            # min-length skip so we exercise the threshold logic directly.
            min_query_chars=0,
        )

    def test_hit_above_threshold(self, tmp_path):
        sem = self._cache(tmp_path)
        base = np.ones(8, dtype=np.float32)
        near = base + np.full(8, 0.01, dtype=np.float32)

        scripter = _EmbeddingScripter(dim=8)
        scripter.register("What is the capital of France?", base)
        scripter.register("Tell me the capital of France.", near)

        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.0):
            sem.put(
                agent_name="a", prompt_hash="p", content_hash="c1",
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                model="m", response_text="Paris",
                input_tokens=10, output_tokens=2,
            )
            hit = sem.get(
                agent_name="a", prompt_hash="p",
                messages=[{"role": "user", "content": "Tell me the capital of France."}],
            )
        assert hit is not None
        assert hit["response_text"] == "Paris"
        assert hit["similarity_score"] >= 0.92

    def test_miss_below_threshold(self, tmp_path):
        sem = self._cache(tmp_path)
        base = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        orthogonal = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        scripter = _EmbeddingScripter(dim=8)
        scripter.register("topic A", base)
        scripter.register("topic B", orthogonal)

        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.0):
            sem.put(
                agent_name="a", prompt_hash="p", content_hash="c1",
                messages=[{"role": "user", "content": "topic A"}],
                model="m", response_text="alpha",
                input_tokens=1, output_tokens=1,
            )
            hit = sem.get(
                agent_name="a", prompt_hash="p",
                messages=[{"role": "user", "content": "topic B"}],
            )
        assert hit is None

    def test_skip_put_if_exact_duplicate_exists(self, tmp_path):
        sem = self._cache(tmp_path)
        scripter = _EmbeddingScripter(dim=8)
        scripter.register("X", np.ones(8, dtype=np.float32))
        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.0):
            for _ in range(3):
                sem.put(
                    agent_name="a", prompt_hash="p", content_hash="c1",
                    messages=[{"role": "user", "content": "X"}],
                    model="m", response_text="R",
                    input_tokens=1, output_tokens=1,
                )
        assert sem.stats()["total_entries"] == 1

    def test_invalidate_agent(self, tmp_path):
        sem = self._cache(tmp_path)
        scripter = _EmbeddingScripter(dim=8)
        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.0):
            sem.put("a", "p", "c1", [{"role": "user", "content": "1"}], "m", "r", 1, 1)
            sem.put("a", "p", "c2", [{"role": "user", "content": "2"}], "m", "r", 1, 1)
            sem.put("b", "p", "c1", [{"role": "user", "content": "3"}], "m", "r", 1, 1)
        deleted = sem.invalidate_agent("a")
        assert deleted == 2
        assert sem.stats()["total_entries"] == 1

    def test_stats_reflect_hits(self, tmp_path):
        sem = self._cache(tmp_path, threshold=0.5)
        scripter = _EmbeddingScripter(dim=8)
        scripter.register("q", np.ones(8, dtype=np.float32))
        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.0):
            sem.put("a", "p", "c1", [{"role": "user", "content": "q"}],
                    "m", "r", 1, 1)
            sem.get("a", "p", [{"role": "user", "content": "q"}])
            sem.get("a", "p", [{"role": "user", "content": "q"}])
        stats = sem.stats()
        assert stats["total_entries"] == 1
        assert stats["total_hits"] == 2

    def test_messages_to_text_skips_system(self):
        text = _messages_to_text([
            {"role": "system", "content": "secret"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ])
        assert "secret" not in text
        assert "hello" in text
        assert "world" in text

    def test_cosine_similarity_edge_cases(self):
        v = np.array([1, 0, 0], dtype=np.float32)
        zeros = np.zeros(3, dtype=np.float32)
        assert _cosine_similarity(v, zeros) == 0.0
        assert _cosine_similarity(v, v) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Broker cascade
# ---------------------------------------------------------------------------


class TestBrokerCascadeWithSemantic:
    def _setup(self, tmp_path, threshold: float = 0.92):
        exact = ExactCache(tmp_path)
        trace_store = TraceStore(tmp_path)
        sem = SemanticCache(
            db_path=exact.db_path,
            embedding_model="text-embedding-3-small",
            threshold=threshold,
            min_query_chars=0,  # short test queries; opt out of cost-control skip
        )
        broker = Broker(
            cache=exact, trace_store=trace_store, cost_tracker=None,
            semantic_cache=sem,
        )
        return broker, exact, sem

    def test_exact_hit_short_circuits_before_semantic(self, tmp_path):
        broker, exact, sem = self._setup(tmp_path)
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        prompt_hash = compute_prompt_hash(messages)
        content_hash = compute_content_hash(messages)
        exact.put("a", prompt_hash, content_hash, "m", "R", 1, 1)

        with patch("litellm.embedding") as embed, \
             patch("thriftai.broker.call_litellm") as live:
            result = broker.route(messages, "m", agent_name="a")
        assert result.resolution is CallResolution.CACHE_HIT
        embed.assert_not_called()
        live.assert_not_called()

    def test_semantic_hit_short_circuits_before_live(self, tmp_path):
        broker, _, sem = self._setup(tmp_path)
        stored_vec = np.ones(8, dtype=np.float32)
        near_vec = stored_vec + 0.01
        scripter = _EmbeddingScripter(dim=8)
        scripter.register("original", stored_vec)
        scripter.register("paraphrase", near_vec)

        # Seed the semantic cache directly.
        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.0):
            sem.put(
                agent_name="a", prompt_hash=compute_prompt_hash(
                    [{"role": "system", "content": "S"}]
                ),
                content_hash="c1",
                messages=[{"role": "user", "content": "original"}],
                model="m", response_text="Paris",
                input_tokens=10, output_tokens=2,
            )

        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "paraphrase"},
        ]
        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.00002), \
             patch("thriftai.broker.call_litellm") as live:
            result = broker.route(messages, "m", agent_name="a")
        assert result.resolution is CallResolution.SEMANTIC_HIT
        assert result.response_text == "Paris"
        assert result.similarity_score is not None and result.similarity_score >= 0.92
        assert result.embedding_cost_usd > 0  # attributed to this call
        live.assert_not_called()

    def test_live_call_populates_both_caches(self, tmp_path):
        broker, exact, sem = self._setup(tmp_path)
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "new query"},
        ]
        scripter = _EmbeddingScripter(dim=8)

        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.00003), \
             patch("thriftai.broker.call_litellm") as live:
            live.return_value = MagicMock(
                response_text="R", model="m",
                input_tokens=10, output_tokens=5, cost_usd=0.05,
            )
            result = broker.route(messages, "m", agent_name="a")

        assert result.resolution is CallResolution.LIVE
        # Embedding cost attributed even on a semantic miss.
        assert result.embedding_cost_usd > 0
        # Exact cache populated
        prompt_hash = compute_prompt_hash(messages)
        content_hash = compute_content_hash(messages)
        assert exact.get("a", prompt_hash, content_hash) is not None
        # Semantic cache populated
        assert sem.stats()["total_entries"] == 1

    def test_no_semantic_cache_means_no_embedding(self, tmp_path):
        """Broker without semantic cache never calls litellm.embedding."""
        exact = ExactCache(tmp_path)
        broker = Broker(
            cache=exact,
            trace_store=TraceStore(tmp_path),
            cost_tracker=None,
            semantic_cache=None,
        )
        messages = [{"role": "user", "content": "U"}]
        with patch("litellm.embedding") as embed, \
             patch("thriftai.broker.call_litellm") as live:
            live.return_value = MagicMock(
                response_text="R", model="m",
                input_tokens=1, output_tokens=1, cost_usd=0.01,
            )
            broker.route(messages, "m", agent_name="a")
        embed.assert_not_called()


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


class TestSessionSemantic:
    def test_semantic_disabled_by_default(self, tmp_path):
        session = Session(cache_dir=tmp_path)
        assert session.semantic_cache is None
        assert session.broker.semantic_cache is None

    def test_semantic_enabled_when_embedding_model_set(self, tmp_path):
        session = Session(
            cache_dir=tmp_path,
            embedding_model="text-embedding-3-small",
            semantic_threshold=0.85,
        )
        assert session.semantic_cache is not None
        assert session.semantic_cache.threshold == 0.85
        assert session.broker.semantic_cache is session.semantic_cache

    def test_cost_report_includes_embedding_cost(self, tmp_path):
        @agent(name="researcher")
        def research(session, topic):
            return session.completion(
                messages=[
                    {"role": "system", "content": "You research."},
                    {"role": "user", "content": topic},
                ],
                model="m",
            )

        session = Session(
            cache_dir=tmp_path, embedding_model="text-embedding-3-small",
        )
        scripter = _EmbeddingScripter(dim=8)

        with patch("litellm.embedding", side_effect=scripter), \
             patch("litellm.completion_cost", return_value=0.00005), \
             patch("thriftai.broker.call_litellm") as live:
            live.return_value = MagicMock(
                response_text="R", model="m",
                input_tokens=10, output_tokens=5, cost_usd=0.05,
            )
            with session.run() as run:
                research(run, "AI costs")

        report = run.cost_report
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.resolution == "live"
        assert entry.embedding_cost_usd > 0
        # Summary should mention the embedding cost line
        text = report.summary()
        assert "Embeddings" in text
