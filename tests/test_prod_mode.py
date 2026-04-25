"""
Production-mode tests for `Session(enabled=False)` and `THRIFTAI_DISABLED`.

What we're protecting:

1. Disabled sessions don't touch the filesystem (so it's safe to bring up
   in environments where the working directory is read-only or where you
   simply don't want artifacts written).
2. `cache.get/put` are no-ops, so every call resolves LIVE.
3. `Session.replay()` raises a clear error — there's no trace to load.
4. Cost tracking still works (the value users keep when they turn caching off).
5. The `THRIFTAI_DISABLED=1` env var **wins over** an explicit `enabled=True`.
6. Disabling is non-destructive: re-enabling the same `cache_dir` later
   sees the original cache contents.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from thriftai.agent import _agent_registry, agent
from thriftai.cache import ExactCache
from thriftai.session import Session, _NoOpCache


@pytest.fixture(autouse=True)
def _reset_registry():
    _agent_registry.clear()
    yield
    _agent_registry.clear()


# ---------------------------------------------------------------------------
# Construction: no filesystem touch
# ---------------------------------------------------------------------------


def test_disabled_session_does_not_touch_filesystem(tmp_path):
    cache_dir = tmp_path / "thriftai-prod"
    assert not cache_dir.exists()

    session = Session(cache_dir=cache_dir, enabled=False)

    # No directory created, no SQLite file, no traces dir.
    assert not cache_dir.exists()
    assert isinstance(session.cache, _NoOpCache)
    assert session.trace_store is None
    assert session.semantic_cache is None
    assert session.enabled is False


def test_enabled_session_does_create_filesystem(tmp_path):
    cache_dir = tmp_path / "thriftai-dev"
    Session(cache_dir=cache_dir, enabled=True)
    assert cache_dir.exists()
    assert (cache_dir / "cache.db").exists()


# ---------------------------------------------------------------------------
# Cache no-op behavior
# ---------------------------------------------------------------------------


def test_disabled_cache_always_misses(tmp_path):
    session = Session(cache_dir=tmp_path, enabled=False)
    assert session.cache.get("a", "p", "c") is None
    # put() must be a silent no-op (no exception, no return value to check)
    session.cache.put(
        agent_name="a", prompt_hash="p", content_hash="c",
        model="m", response_text="x", input_tokens=1, output_tokens=1,
    )
    assert session.cache.get("a", "p", "c") is None
    # invalidate returns 0 (nothing to delete)
    assert session.cache.invalidate_agent("a") == 0
    # stats are zero-valued but well-formed
    stats = session.cache.stats()
    assert stats["total_entries"] == 0
    assert stats["total_hits"] == 0


def test_disabled_run_routes_every_call_live(tmp_path):
    @agent(name="r")
    def r(s):
        return s.completion(
            messages=[
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
            ],
            model="m",
        )

    session = Session(cache_dir=tmp_path, enabled=False)
    fake = MagicMock(
        response_text="R", model="m",
        input_tokens=10, output_tokens=5, cost_usd=0.05,
    )
    with patch("thriftai.broker.call_litellm", side_effect=[fake, fake, fake]):
        with session.run() as run:
            r(run)
            r(run)
            r(run)

    # Three calls → three live invocations recorded in the cost report.
    assert len(run.cost_report.entries) == 3
    assert all(e.resolution == "live" for e in run.cost_report.entries)
    # Cost tracking still works.
    assert run.cost_report.total_cost == pytest.approx(0.15)


def test_disabled_run_writes_no_trace_files(tmp_path):
    @agent(name="r")
    def r(s):
        return s.completion(
            messages=[{"role": "user", "content": "U"}], model="m",
        )

    session = Session(cache_dir=tmp_path, enabled=False)
    with patch("thriftai.broker.call_litellm",
               return_value=MagicMock(
                   response_text="R", model="m",
                   input_tokens=1, output_tokens=1, cost_usd=0.01,
               )):
        with session.run() as run:
            r(run)

    # No .thriftai dir means no traces dir means no JSON files.
    assert not (tmp_path / "traces").exists()


# ---------------------------------------------------------------------------
# Replay guard
# ---------------------------------------------------------------------------


def test_replay_raises_when_disabled(tmp_path):
    session = Session(cache_dir=tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        session.replay(trace_id="anything")


# ---------------------------------------------------------------------------
# Env var precedence
# ---------------------------------------------------------------------------


def test_env_var_disables_even_when_kwarg_says_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("THRIFTAI_DISABLED", "1")
    session = Session(cache_dir=tmp_path, enabled=True)
    assert session.enabled is False
    assert isinstance(session.cache, _NoOpCache)


def test_env_var_only_disables_on_exact_value_1(tmp_path, monkeypatch):
    # Only the literal "1" disables; "true", "yes", etc. don't.
    monkeypatch.setenv("THRIFTAI_DISABLED", "true")
    session = Session(cache_dir=tmp_path)
    assert session.enabled is True


def test_no_env_var_defaults_to_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("THRIFTAI_DISABLED", raising=False)
    session = Session(cache_dir=tmp_path)
    assert session.enabled is True


# ---------------------------------------------------------------------------
# Non-destructive: enable → disable → enable preserves cache contents
# ---------------------------------------------------------------------------


def test_disabling_then_reenabling_preserves_cache(tmp_path):
    # Phase 1: enabled, populate the cache via direct API (skip the broker).
    s1 = Session(cache_dir=tmp_path, enabled=True)
    assert isinstance(s1.cache, ExactCache)
    s1.cache.put(
        agent_name="r", prompt_hash="p", content_hash="c",
        model="m", response_text="hello", input_tokens=1, output_tokens=1,
    )
    assert s1.cache.get("r", "p", "c")["response_text"] == "hello"

    # Phase 2: disabled — same cache_dir, no destructive side effects.
    s2 = Session(cache_dir=tmp_path, enabled=False)
    # The disabled cache reports nothing — but the SQLite file is intact.
    assert s2.cache.get("r", "p", "c") is None
    assert (tmp_path / "cache.db").exists()

    # Phase 3: re-enable — the original entry is still there.
    s3 = Session(cache_dir=tmp_path, enabled=True)
    hit = s3.cache.get("r", "p", "c")
    assert hit is not None
    assert hit["response_text"] == "hello"


# ---------------------------------------------------------------------------
# Semantic cache disabled regardless of embedding_model
# ---------------------------------------------------------------------------


def test_disabled_session_ignores_embedding_model(tmp_path):
    """Even with a model passed, semantic cache stays off when disabled."""
    session = Session(
        cache_dir=tmp_path,
        enabled=False,
        embedding_model="text-embedding-3-small",
    )
    assert session.semantic_cache is None
    assert session.broker.semantic_cache is None
