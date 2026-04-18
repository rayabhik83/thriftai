"""
Semantic cache — embedding-based approximate matching.

Complements the exact cache. When exact match misses, the semantic cache
checks whether a previous response for a *similar* input can be reused.

Key policy:
- Cache key scope is (agent_name, prompt_hash). An entry is only eligible
  when the system prompt matches exactly; only the user/tool content can
  vary across entries.
- We skip storing embeddings for (agent_name, prompt_hash, content_hash)
  triples that the exact cache already covers — the exact cache wins.
- Similarity is cosine; threshold is configurable (default 0.92).

Storage sits in the same SQLite file as the exact cache. Embeddings are
stored as raw float32 bytes (np.ndarray.tobytes()); dimension is inferred
on read from blob length (so we aren't tied to any one embedding model).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _messages_to_text(messages: list[dict]) -> str:
    """Flatten non-system messages into a single string for embedding."""
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            # Anthropic-style content blocks: concatenate text parts.
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        parts.append(str(content))
    return "\n".join(parts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class SemanticCache:
    """Embedding-based approximate cache for near-miss inputs."""

    def __init__(
        self,
        db_path: Path,
        embedding_model: str,
        threshold: float = 0.92,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # See cache/__init__.py for the rationale — sqlite3 needs serialized
        # access per-connection. The embedding network call is explicitly
        # kept outside the lock to avoid blocking other cache ops on I/O.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # Tracks the $ spent on embedding API calls since the last reset.
        # The broker reads and resets this per call so the cost report can
        # attribute embedding cost to the correct agent.
        self._cost_lock = threading.Lock()
        self._pending_embed_cost_usd: float = 0.0

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_agent_prompt
                ON semantic_cache(agent_name, prompt_hash)
                """
            )
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_dedupe
                ON semantic_cache(agent_name, prompt_hash, content_hash)
                """
            )
            self._conn.commit()

    # -- embedding --------------------------------------------------------

    def embed(self, messages: list[dict]) -> np.ndarray:
        """Embed non-system messages via LiteLLM.

        Side effect: accumulates API cost into `_pending_embed_cost_usd`.
        Network I/O is intentionally *outside* the DB lock so other cache
        operations don't serialize behind embedding round-trips.
        """
        import litellm

        text = _messages_to_text(messages)
        response = litellm.embedding(model=self.embedding_model, input=[text])
        vector = np.asarray(response.data[0]["embedding"], dtype=np.float32)
        cost = _embedding_response_cost(response)
        with self._cost_lock:
            self._pending_embed_cost_usd += cost
        return vector

    def take_pending_embed_cost(self) -> float:
        """Return accumulated embedding cost and reset the counter."""
        with self._cost_lock:
            cost = self._pending_embed_cost_usd
            self._pending_embed_cost_usd = 0.0
        return cost

    # -- public API -------------------------------------------------------

    def get(
        self,
        agent_name: str,
        prompt_hash: str,
        messages: list[dict],
        *,
        precomputed_embedding: np.ndarray | None = None,
    ) -> dict | None:
        """Return the best matching entry if similarity >= threshold, else None.

        If `precomputed_embedding` is given, we skip re-embedding (used by
        the broker when it wants to reuse the same vector for get + put).
        """
        query = (
            precomputed_embedding
            if precomputed_embedding is not None
            else self.embed(messages)
        )
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, embedding, model, response_text, input_tokens, output_tokens
                FROM semantic_cache
                WHERE agent_name = ? AND prompt_hash = ?
                """,
                (agent_name, prompt_hash),
            ).fetchall()

        if not rows:
            return None

        best: sqlite3.Row | None = None
        best_sim = -1.0
        for row in rows:
            stored = _deserialize_embedding(row["embedding"])
            if stored.shape != query.shape:
                # Embedding dim mismatch (likely user swapped the model).
                # Skip this row rather than crash.
                continue
            sim = _cosine_similarity(query, stored)
            if sim > best_sim:
                best_sim = sim
                best = row

        if best is None or best_sim < self.threshold:
            return None

        with self._lock:
            self._conn.execute(
                "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = ?",
                (best["id"],),
            )
            self._conn.commit()

        return {
            "response_text": best["response_text"],
            "model": best["model"],
            "input_tokens": best["input_tokens"],
            "output_tokens": best["output_tokens"],
            "similarity_score": best_sim,
        }

    def put(
        self,
        agent_name: str,
        prompt_hash: str,
        content_hash: str,
        messages: list[dict],
        model: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        *,
        precomputed_embedding: np.ndarray | None = None,
    ) -> None:
        """Store a response with its embedding.

        Skipped if this exact (agent_name, prompt_hash, content_hash) is
        already present — the exact cache handles those lookups.
        """
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT 1 FROM semantic_cache
                WHERE agent_name = ? AND prompt_hash = ? AND content_hash = ?
                LIMIT 1
                """,
                (agent_name, prompt_hash, content_hash),
            ).fetchone()
        if existing is not None:
            return

        vector = (
            precomputed_embedding
            if precomputed_embedding is not None
            else self.embed(messages)
        )
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO semantic_cache
                    (agent_name, prompt_hash, content_hash, embedding, model,
                     response_text, input_tokens, output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_name,
                    prompt_hash,
                    content_hash,
                    vector.tobytes(),
                    model,
                    response_text,
                    input_tokens,
                    output_tokens,
                ),
            )
            self._conn.commit()

    def invalidate_agent(self, agent_name: str) -> int:
        """Delete all semantic cache entries for an agent."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM semantic_cache WHERE agent_name = ?", (agent_name,)
            )
            self._conn.commit()
            return cursor.rowcount

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n,
                       COALESCE(SUM(hit_count), 0) AS hits
                FROM semantic_cache
                """
            ).fetchone()
        return {
            "total_entries": int(row["n"]),
            "total_hits": int(row["hits"]),
            "threshold": self.threshold,
            "embedding_model": self.embedding_model,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _embedding_response_cost(response: Any) -> float:
    """Best-effort cost extraction from a LiteLLM embedding response.

    LiteLLM doesn't expose a dedicated `embedding_cost()` helper, but
    `completion_cost(completion_response=...)` handles embedding responses
    in recent versions. Fall back to 0 on any error so the cache still
    works even when cost data isn't available for the model.
    """
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception as e:  # pragma: no cover — cost data is best-effort
        log.debug("embedding cost lookup failed: %s", e)
        return 0.0
