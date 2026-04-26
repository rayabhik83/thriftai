"""
Scale smoke tests for the SemanticCache.

Skipped from default CI (`pytest -m slow` to opt in). These take a couple
of seconds each and aren't worth running on every PR; they exist to catch
*regressions* in the linear-scan + bucket-cap behavior at realistic
working-set sizes.

Numbers asserted are conservative — they pass on a 2024-vintage laptop and
should pass on any reasonable CI runner. If they fail in CI on a smaller
runner, raise the thresholds rather than special-casing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from thriftai.cache.semantic import SemanticCache


pytestmark = pytest.mark.slow


def _seed_bucket(sem: SemanticCache, n: int, dim: int = 1536) -> None:
    """Seed one bucket with `n` distinct entries via direct DB writes —
    bypasses the bucket-cap eviction so we can measure scan time."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        v = rng.standard_normal(dim).astype(np.float32).tobytes()
        rows.append(("ag", "p", f"c{i:06d}", v, "m", f"r{i}", 1, 1))
    with sem._lock:
        sem._conn.executemany(
            """
            INSERT INTO semantic_cache
                (agent_name, prompt_hash, content_hash, embedding, model,
                 response_text, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        sem._conn.commit()


def test_lookup_p95_under_100ms_at_10k_entries(tmp_path):
    """Linear similarity scan over 10k entries × 1536-dim should stay
    under 100 ms p95 on the local machine."""
    sem = SemanticCache(
        db_path=tmp_path / "cache.db",
        embedding_model="m",
        threshold=0.92,
        min_query_chars=0,
        bucket_size=10_001,  # avoid eviction during seeding
    )
    _seed_bucket(sem, n=10_000)

    rng = np.random.default_rng(1)
    query = rng.standard_normal(1536).astype(np.float32)

    timings: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        sem.get(
            agent_name="ag", prompt_hash="p",
            messages=[{"role": "user", "content": "x" * 200}],
            precomputed_embedding=query,
        )
        timings.append(time.perf_counter() - t0)

    timings.sort()
    p95 = timings[int(len(timings) * 0.95)]
    assert p95 < 0.100, f"p95 lookup at 10k entries was {p95*1000:.1f} ms (>100 ms)"


def test_stats_under_500ms_at_100k_entries(tmp_path):
    """`stats()` is a COUNT/SUM aggregate — should stay fast even at
    100k entries spread across many buckets."""
    sem = SemanticCache(
        db_path=tmp_path / "cache.db",
        embedding_model="m",
        threshold=0.92,
        min_query_chars=0,
        bucket_size=10_000,
    )
    rng = np.random.default_rng(0)
    rows = []
    for bucket in range(100):
        for i in range(1000):
            v = rng.standard_normal(384).astype(np.float32).tobytes()
            rows.append((f"ag{bucket}", f"p{bucket}", f"c{i:06d}", v, "m", "r", 1, 1))
    with sem._lock:
        sem._conn.executemany(
            """
            INSERT INTO semantic_cache
                (agent_name, prompt_hash, content_hash, embedding, model,
                 response_text, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        sem._conn.commit()

    t0 = time.perf_counter()
    stats = sem.stats()
    elapsed = time.perf_counter() - t0
    assert stats["total_entries"] == 100_000
    assert elapsed < 0.500, f"stats() at 100k entries was {elapsed*1000:.1f} ms (>500 ms)"
