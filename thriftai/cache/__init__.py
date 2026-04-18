"""
Exact-match cache using content hashing.

Cache key: (agent_name, prompt_template_hash, content_hash)
- agent_name: from the @agent decorator's thread-local
- prompt_template_hash: SHA-256 of the system message content
- content_hash: SHA-256 of all non-system messages serialized

Two agents sending identical messages do NOT share cache entries,
because agent_name is part of the key. This is correct — the same
input to an observer vs. a synthesizer should produce different outputs.

Storage backend: SQLite (single file, zero config, portable).

Schema:
    cache_entries (
        key TEXT PRIMARY KEY,     -- composite hash
        agent_name TEXT,
        prompt_hash TEXT,
        content_hash TEXT,
        model TEXT,
        response_text TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        created_at TIMESTAMP,
        hit_count INTEGER DEFAULT 0
    )
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def compute_prompt_hash(messages: list[dict]) -> str:
    """Hash the system message(s) to detect prompt template changes."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    content = json.dumps(system_msgs, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_content_hash(messages: list[dict]) -> str:
    """Hash non-system messages to detect input changes."""
    non_system = [m for m in messages if m.get("role") != "system"]
    content = json.dumps(non_system, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _composite_key(agent_name: str, prompt_hash: str, content_hash: str) -> str:
    raw = f"{agent_name}|{prompt_hash}|{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()


class ExactCache:
    """SQLite-backed exact-match response cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        # Python 3.14's sqlite3 rejects concurrent use of a single connection
        # from multiple threads even with check_same_thread=False. Guard every
        # DB op with a reentrant lock so the cache is safe to share across a
        # multi-threaded agent framework.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_name ON cache_entries(agent_name)"
            )
            self._conn.commit()

    def get(self, agent_name: str, prompt_hash: str, content_hash: str) -> dict | None:
        """Look up a cached response. Returns None on miss."""
        key = _composite_key(agent_name, prompt_hash, content_hash)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE key = ?",
                (key,),
            )
            self._conn.commit()
            return dict(row)

    def put(
        self,
        agent_name: str,
        prompt_hash: str,
        content_hash: str,
        model: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Store a response in the cache."""
        key = _composite_key(agent_name, prompt_hash, content_hash)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                    (key, agent_name, prompt_hash, content_hash, model,
                     response_text, input_tokens, output_tokens, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    key,
                    agent_name,
                    prompt_hash,
                    content_hash,
                    model,
                    response_text,
                    input_tokens,
                    output_tokens,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def invalidate_agent(self, agent_name: str) -> int:
        """Invalidate all cache entries for an agent. Returns count deleted."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE agent_name = ?", (agent_name,)
            )
            self._conn.commit()
            return cursor.rowcount

    def stats(self) -> dict:
        """Return cache statistics: total entries, hit counts, size."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(hit_count), 0) AS hits FROM cache_entries"
            ).fetchone()
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "total_entries": int(row["n"]),
            "total_hits": int(row["hits"]),
            "db_size_bytes": size,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
