"""
Recovery / corrupted-state tests.

Real-world failure modes a thriftai user will eventually hit:

- Process killed mid-write → trace JSON file is truncated.
- Cache directory wiped between runs → ExactCache rebuilds cleanly.
- Library upgraded across a schema change → on-disk DB no longer matches.
- Trace JSON hand-edited and broken.
- replay() called for a trace_id that doesn't exist.

These should all produce **clear, actionable errors** — not silent garbage,
not bare `JSONDecodeError`s.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from thriftai.cache import SCHEMA_VERSION, ExactCache, SchemaVersionError
from thriftai.session import Session
from thriftai.trace import Trace, TraceEntry, TraceStore


# ---------------------------------------------------------------------------
# ExactCache: missing DB / fresh init
# ---------------------------------------------------------------------------


def test_missing_cache_dir_is_created(tmp_path):
    """ExactCache pointed at a non-existent dir creates it on init."""
    target = tmp_path / "fresh"
    assert not target.exists()
    cache = ExactCache(target)
    assert target.exists()
    assert (target / "cache.db").exists()
    # Empty cache returns None on lookup, no exceptions.
    assert cache.get("a", "p", "c") is None


def test_empty_table_misses_cleanly(tmp_path):
    """An ExactCache with no entries returns None on get and 0 on stats."""
    cache = ExactCache(tmp_path)
    assert cache.get("a", "p", "c") is None
    stats = cache.stats()
    assert stats["total_entries"] == 0
    assert stats["total_hits"] == 0


# ---------------------------------------------------------------------------
# Schema-version mismatch
# ---------------------------------------------------------------------------


def test_fresh_db_gets_current_schema_version(tmp_path):
    cache = ExactCache(tmp_path)
    user_version = cache._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == SCHEMA_VERSION


def test_existing_db_at_correct_version_opens_cleanly(tmp_path):
    # First open: stamps SCHEMA_VERSION.
    ExactCache(tmp_path)
    # Second open: reads back the version, must not raise.
    ExactCache(tmp_path)


def test_existing_db_at_higher_version_raises(tmp_path):
    """Simulate a forward-incompatible DB: a future version of thriftai
    wrote it, now we're running an older one. Must raise, not corrupt."""
    # Create a DB and stamp it with a higher version.
    db = tmp_path / "cache.db"
    conn = sqlite3.connect(str(db))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 99}")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError) as excinfo:
        ExactCache(tmp_path)
    msg = str(excinfo.value)
    assert "schema version" in msg
    assert str(SCHEMA_VERSION + 99) in msg
    assert "Delete the cache directory" in msg


def test_existing_db_at_older_version_raises(tmp_path):
    """A user upgrades thriftai across a schema change. Old DB → must raise."""
    if SCHEMA_VERSION <= 1:
        pytest.skip("no older version exists yet (SCHEMA_VERSION == 1)")
    db = tmp_path / "cache.db"
    conn = sqlite3.connect(str(db))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        ExactCache(tmp_path)


# ---------------------------------------------------------------------------
# TraceStore: missing / truncated / malformed traces
# ---------------------------------------------------------------------------


def test_missing_trace_raises_filenotfound(tmp_path):
    store = TraceStore(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        store.load("does_not_exist")
    assert "does_not_exist" in str(excinfo.value)


def test_truncated_trace_json_raises_value_error(tmp_path):
    """Simulate a Ctrl-C during record(): file exists but is truncated.
    load() must surface a clear error mentioning the trace_id and likely cause."""
    store = TraceStore(tmp_path)
    bad_path = store.traces_dir / "broken.json"
    bad_path.write_text('{"trace_id": "broken", "entries": [{"sequen')  # truncated mid-key

    with pytest.raises(ValueError) as excinfo:
        store.load("broken")
    msg = str(excinfo.value)
    assert "broken.json" in msg
    assert "not valid JSON" in msg
    # Original JSONDecodeError chained for debugging
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_trace_missing_required_fields_raises_value_error(tmp_path):
    """Hand-edited trace file missing trace_id key — must raise with context."""
    store = TraceStore(tmp_path)
    bad_path = store.traces_dir / "incomplete.json"
    bad_path.write_text('{"entries": []}')  # no trace_id

    with pytest.raises(ValueError) as excinfo:
        store.load("incomplete")
    assert "incomplete.json" in str(excinfo.value)
    assert "Delete the file" in str(excinfo.value)


def test_trace_with_extra_fields_loads_anyway(tmp_path):
    """A future thriftai version may add fields. Older versions should
    ignore unknown keys rather than crash, as long as required keys exist."""
    store = TraceStore(tmp_path)
    path = store.traces_dir / "future.json"
    path.write_text(json.dumps({
        "trace_id": "future",
        "created_at": "2026-04-26T00:00:00Z",
        "entries": [],
        "total_cost_usd": 0.0,
        # New field a future version added:
        "experimental_score": 0.99,
    }))
    # Should NOT raise — we ignore unknown top-level fields.
    trace = store.load("future")
    assert trace.trace_id == "future"


def test_trace_round_trip_after_recovery(tmp_path):
    """End-to-end: after a corrupted file is detected and removed, the
    cache stays usable for new traces."""
    store = TraceStore(tmp_path)
    (store.traces_dir / "broken.json").write_text("not json")
    with pytest.raises(ValueError):
        store.load("broken")
    # Cleanup what the user would do
    (store.traces_dir / "broken.json").unlink()

    # Fresh trace lands cleanly
    t = Trace(trace_id="ok", entries=[
        TraceEntry(sequence=0, agent_name="a", model="m",
                   messages_hash="h", response_text="r")
    ])
    store.record(t)
    loaded = store.load("ok")
    assert loaded.trace_id == "ok"
    assert len(loaded.entries) == 1


# ---------------------------------------------------------------------------
# Session.replay() error surface
# ---------------------------------------------------------------------------


def test_session_replay_missing_trace_raises_filenotfound(tmp_path):
    session = Session(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        session.replay(trace_id="never_recorded")


def test_session_replay_truncated_trace_raises_value_error(tmp_path):
    session = Session(cache_dir=tmp_path)
    # Drop a corrupted file directly into the traces dir.
    assert session.trace_store is not None
    (session.trace_store.traces_dir / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        session.replay(trace_id="bad")
