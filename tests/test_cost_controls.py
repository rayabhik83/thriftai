"""
Tests for the SemanticCache cost-control knobs:

- `min_query_chars` — short queries skip the embedding round-trip entirely
  (no API call, no DB write). Saves money when the LLM call is cheap
  enough that embedding it would be a net loss.
- `bucket_size` — entries per `(agent_name, prompt_hash)` are FIFO-evicted
  when the cap is exceeded. Keeps the linear similarity scan bounded.

Both are exposed at the `Session` level too.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thriftai.cache.semantic import SemanticCache
from thriftai.session import Session


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestParamValidation:
    def test_negative_min_query_chars_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="min_query_chars"):
            SemanticCache(
                db_path=tmp_path / "cache.db",
                embedding_model="m",
                min_query_chars=-1,
            )

    def test_bucket_size_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="bucket_size"):
            SemanticCache(
                db_path=tmp_path / "cache.db",
                embedding_model="m",
                bucket_size=0,
            )


# ---------------------------------------------------------------------------
# min_query_chars: skip embed for short queries
# ---------------------------------------------------------------------------


class TestMinQueryChars:
    def _cache(self, tmp_path, *, min_chars: int):
        return SemanticCache(
            db_path=tmp_path / "cache.db",
            embedding_model="m",
            threshold=0.92,
            min_query_chars=min_chars,
        )

    def test_get_skips_embed_when_text_below_min(self, tmp_path):
        sem = self._cache(tmp_path, min_chars=100)
        with patch("litellm.embedding") as embed:
            result = sem.get(
                agent_name="a", prompt_hash="p",
                messages=[{"role": "user", "content": "short"}],  # 5 chars
            )
        assert result is None
        embed.assert_not_called()

    def test_put_skips_embed_when_text_below_min(self, tmp_path):
        sem = self._cache(tmp_path, min_chars=100)
        with patch("litellm.embedding") as embed:
            sem.put(
                agent_name="a", prompt_hash="p", content_hash="c",
                messages=[{"role": "user", "content": "short"}],
                model="m", response_text="r", input_tokens=1, output_tokens=1,
            )
        assert sem.stats()["total_entries"] == 0
        embed.assert_not_called()

    def test_get_does_embed_when_text_above_min(self, tmp_path):
        sem = self._cache(tmp_path, min_chars=10)
        long_text = "this is definitely longer than ten characters"
        scripted = MagicMock()
        scripted.data = [{"embedding": np.zeros(8, dtype=np.float32).tolist()}]
        with patch("litellm.embedding", return_value=scripted) as embed, \
             patch("litellm.completion_cost", return_value=0.0):
            sem.get(
                agent_name="a", prompt_hash="p",
                messages=[{"role": "user", "content": long_text}],
            )
        embed.assert_called_once()

    def test_min_chars_zero_disables_skip(self, tmp_path):
        sem = self._cache(tmp_path, min_chars=0)
        scripted = MagicMock()
        scripted.data = [{"embedding": np.zeros(8, dtype=np.float32).tolist()}]
        with patch("litellm.embedding", return_value=scripted) as embed, \
             patch("litellm.completion_cost", return_value=0.0):
            sem.get(
                agent_name="a", prompt_hash="p",
                messages=[{"role": "user", "content": "x"}],  # 1 char
            )
        embed.assert_called_once()

    def test_precomputed_embedding_bypasses_skip(self, tmp_path):
        """If the broker hands us a pre-computed vector, we use it
        regardless of length — the embedding cost was already paid upstream."""
        sem = self._cache(tmp_path, min_chars=100)
        precomputed = np.ones(8, dtype=np.float32)
        # Seed a matching entry directly via put with a precomputed vector
        # so the short-query skip on put doesn't fire.
        sem.put(
            agent_name="a", prompt_hash="p", content_hash="c",
            messages=[{"role": "user", "content": "short"}],
            model="m", response_text="hi",
            input_tokens=1, output_tokens=1,
            precomputed_embedding=precomputed,
        )
        with patch("litellm.embedding") as embed:
            hit = sem.get(
                agent_name="a", prompt_hash="p",
                messages=[{"role": "user", "content": "short"}],
                precomputed_embedding=precomputed,
            )
        assert hit is not None and hit["response_text"] == "hi"
        embed.assert_not_called()


# ---------------------------------------------------------------------------
# bucket_size: FIFO eviction
# ---------------------------------------------------------------------------


def _seed(sem: SemanticCache, agent: str, prompt_hash: str, n: int):
    """Seed `n` entries into one bucket with monotonically increasing
    content_hashes so they're trivially distinguishable, and unique
    embeddings so the dedupe-on-content_hash check doesn't fire."""
    for i in range(n):
        # Embedding has to differ per entry, otherwise the unique index on
        # (agent, prompt, content_hash) would still let multiple inserts
        # land — but their vectors would all be identical.
        # Different content_hash makes each row distinct; the vector content
        # itself doesn't matter for FIFO testing.
        v = np.full(8, fill_value=float(i + 1), dtype=np.float32)
        sem.put(
            agent_name=agent, prompt_hash=prompt_hash,
            content_hash=f"c{i:05d}",
            messages=[{"role": "user", "content": f"msg-{i}"}],
            model="m", response_text=f"r{i}",
            input_tokens=1, output_tokens=1,
            precomputed_embedding=v,
        )


class TestBucketEviction:
    def test_bucket_caps_at_size(self, tmp_path):
        sem = SemanticCache(
            db_path=tmp_path / "cache.db",
            embedding_model="m",
            min_query_chars=0,
            bucket_size=10,
        )
        _seed(sem, "a", "p", n=15)
        # Count entries in the bucket directly via the underlying connection
        n = sem._conn.execute(
            "SELECT COUNT(*) AS n FROM semantic_cache "
            "WHERE agent_name = ? AND prompt_hash = ?",
            ("a", "p"),
        ).fetchone()["n"]
        assert n == 10

    def test_oldest_entries_are_dropped(self, tmp_path):
        sem = SemanticCache(
            db_path=tmp_path / "cache.db",
            embedding_model="m",
            min_query_chars=0,
            bucket_size=5,
        )
        _seed(sem, "a", "p", n=8)
        # FIFO: c00000..c00002 should have been evicted, c00003..c00007 survive
        rows = sem._conn.execute(
            "SELECT content_hash FROM semantic_cache "
            "WHERE agent_name = ? AND prompt_hash = ? ORDER BY content_hash",
            ("a", "p"),
        ).fetchall()
        surviving = [r["content_hash"] for r in rows]
        assert surviving == [f"c{i:05d}" for i in range(3, 8)]

    def test_eviction_does_not_leak_across_buckets(self, tmp_path):
        sem = SemanticCache(
            db_path=tmp_path / "cache.db",
            embedding_model="m",
            min_query_chars=0,
            bucket_size=5,
        )
        # Two independent buckets, both filled past the cap.
        _seed(sem, "a", "p1", n=8)
        _seed(sem, "a", "p2", n=8)
        for prompt in ("p1", "p2"):
            n = sem._conn.execute(
                "SELECT COUNT(*) AS n FROM semantic_cache "
                "WHERE agent_name = ? AND prompt_hash = ?",
                ("a", prompt),
            ).fetchone()["n"]
            assert n == 5, f"bucket {prompt} count={n}"


# ---------------------------------------------------------------------------
# Session-level passthrough
# ---------------------------------------------------------------------------


class TestSessionPlumbing:
    def test_session_passes_kwargs_to_semantic_cache(self, tmp_path):
        session = Session(
            cache_dir=tmp_path,
            embedding_model="text-embedding-3-small",
            semantic_min_chars=42,
            semantic_bucket_size=7,
        )
        assert session.semantic_cache is not None
        assert session.semantic_cache.min_query_chars == 42
        assert session.semantic_cache.bucket_size == 7

    def test_session_defaults(self, tmp_path):
        session = Session(
            cache_dir=tmp_path,
            embedding_model="text-embedding-3-small",
        )
        assert session.semantic_cache is not None
        # Defaults match the SemanticCache defaults.
        assert session.semantic_cache.min_query_chars == 100
        assert session.semantic_cache.bucket_size == 1000
