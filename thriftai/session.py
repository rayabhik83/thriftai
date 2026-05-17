"""
Session — the main entry point for ThriftAI.

A Session controls a single pipeline run. It manages:
- Whether calls go live, hit cache, or replay from a trace
- Cost tracking across all agents in the run
- Trace recording for future replay

Usage:
    session = Session(cache_dir="./thriftai_cache")

    # Normal run — calls go live, responses are cached and traced
    with session.run() as run:
        result = run.completion(messages=[...], model="claude-sonnet-4-20250514")

    # Replay run — only the specified agents go live, rest replay from trace
    with session.replay(trace_id="run_043", live=["hypothesizer"]) as run:
        result = run.completion(messages=[...], model="claude-sonnet-4-20250514")

Design notes:
- Session stores config (cache backend, trace storage path, model ladder)
- session.run() returns a RunContext (context manager)
- session.replay() returns a ReplayContext (context manager)
- Both contexts expose .completion() which routes through the Broker
- The active agent is tracked via _current_agent thread-local (set by @agent decorator)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from thriftai.agent import get_current_agent, get_dependents
from thriftai.broker import Broker, CallResolution
from thriftai.cache import ExactCache, compute_content_hash, compute_prompt_hash
from thriftai.cost import AgentCostEntry, CostReport
from thriftai.trace import Trace, TraceEntry, TraceStore

try:  # numpy is only required when semantic cache is enabled
    from thriftai.cache.semantic import SemanticCache
except ImportError:  # pragma: no cover
    SemanticCache = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for a ThriftAI session."""
    cache_dir: Path = Path(".thriftai")
    embedding_model: str | None = None
    semantic_threshold: float = 0.92
    semantic_min_chars: int = 100
    semantic_bucket_size: int = 1000
    enabled: bool = True


def _new_trace_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


class _NoOpCache:
    """Drop-in for ExactCache when ThriftAI is disabled.

    Always misses, never writes, no SQLite handle. The broker's cascade
    falls through cleanly: every call resolves LIVE, but cost tracking
    still works because that lives in CostReport, not the cache.
    """
    db_path: Path | None = None

    def get(self, agent_name: str, prompt_hash: str, content_hash: str) -> dict | None:
        return None

    def put(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    def invalidate_agent(self, agent_name: str) -> int:  # noqa: ARG002
        return 0

    def stats(self) -> dict:
        return {"total_entries": 0, "total_hits": 0, "db_size_bytes": 0}

    def close(self) -> None:
        return None


class Session:
    """The main entry point for ThriftAI.

    A `Session` owns the on-disk cache and trace store, plus the broker that
    routes every LLM call through the replay → cache → live cascade. Create
    one per project or pipeline; reuse it across runs so they share cache and
    trace history.

    The session is also the kill switch. Setting `enabled=False` (or the
    environment variable `THRIFTAI_DISABLED=1`) turns the session into a thin
    pass-through to LiteLLM: no filesystem writes, no embedding calls, no
    traces. Cost tracking still works.

    Args:
        cache_dir: Directory for the SQLite cache, semantic embeddings, and
            trace JSON files. Created if it doesn't exist.
        embedding_model: LiteLLM model name for semantic-cache embeddings
            (e.g. `"text-embedding-3-small"`). Pass `None` to disable semantic
            cache entirely — exact-match cache and replay still work.
        semantic_threshold: Cosine-similarity floor for a semantic-cache hit
            (0.0–1.0). Higher is stricter.
        semantic_min_chars: Minimum query length before semantic lookup runs.
            Below this, semantic cache is skipped and the call falls through
            to live. Guards against cheap-query, cheap-model break-even loss.
        semantic_bucket_size: Max number of cached entries to scan per
            similarity comparison. Caps lookup latency on large caches.
        enabled: Master switch. `False` disables cache + replay (cost
            tracking stays on). `None` defers to `THRIFTAI_DISABLED`. When
            both are set, the env var wins.

    Example:
        ```python
        import thriftai as ta

        session = ta.Session(cache_dir="./.thriftai")

        with session.run() as run:
            result = run.completion(
                messages=[{"role": "user", "content": "hi"}],
                model="anthropic/claude-sonnet-4-20250514",
            )
            print(run.cost_report.summary())
        ```
    """

    def __init__(
        self,
        cache_dir: str | Path = ".thriftai",
        embedding_model: str | None = None,
        semantic_threshold: float = 0.92,
        semantic_min_chars: int = 100,
        semantic_bucket_size: int = 1000,
        enabled: bool | None = None,
    ):
        # THRIFTAI_DISABLED=1 is the global kill switch and wins over the kwarg.
        # Other values (incl. unset) leave the kwarg untouched.
        if os.environ.get("THRIFTAI_DISABLED") == "1":
            enabled = False
        elif enabled is None:
            enabled = True

        self.enabled = enabled
        self.config = SessionConfig(
            cache_dir=Path(cache_dir),
            embedding_model=embedding_model,
            semantic_threshold=semantic_threshold,
            semantic_min_chars=semantic_min_chars,
            semantic_bucket_size=semantic_bucket_size,
            enabled=enabled,
        )

        self.semantic_cache: SemanticCache | None = None
        if self.enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache: ExactCache | _NoOpCache = ExactCache(self.config.cache_dir)
            self.trace_store: TraceStore | None = TraceStore(self.config.cache_dir)
            if embedding_model is not None:
                if SemanticCache is None:
                    raise RuntimeError(
                        "semantic cache requires numpy: "
                        "pip install 'thriftai[semantic]'"
                    )
                self.semantic_cache = SemanticCache(
                    db_path=self.cache.db_path,
                    embedding_model=embedding_model,
                    threshold=semantic_threshold,
                    min_query_chars=semantic_min_chars,
                    bucket_size=semantic_bucket_size,
                )
        else:
            # Disabled: no filesystem touch, no embeddings, no traces.
            # The broker's cascade still runs; it just always falls through
            # to LIVE because every cache.get() returns None.
            self.cache = _NoOpCache()
            self.trace_store = None

        self.broker = Broker(
            cache=self.cache,
            trace_store=self.trace_store,
            cost_tracker=None,
            semantic_cache=self.semantic_cache,
        )

    def run(self, trace_id: str | None = None) -> "RunContext":
        """Start a normal (live) run.

        Every `completion()` inside the context routes through the broker:
        cache hits short-circuit, cache misses go live and are recorded into
        the cache and trace. The trace is written on context exit.

        Args:
            trace_id: Identifier for the new trace. Defaults to a
                timestamped ID like `run_20260516_154523`.

        Returns:
            A `RunContext` to use as a context manager.

        Example:
            ```python
            with session.run() as run:
                out = run.completion(messages=[...], model="...")
                print(run.trace_id)
            ```
        """
        return RunContext(session=self, trace_id=trace_id or _new_trace_id())

    def replay(
        self, trace_id: str, live: list[str] | None = None
    ) -> "ReplayContext":
        """Replay a previous run, sending only selected agents live.

        Agents listed in `live` go through the normal cache → live cascade.
        Every other agent's call is served from the recorded trace.

        If a live agent's output differs from what was recorded in the trace,
        all of its transitive dependents are invalidated for the rest of the
        replay: they skip the replay path and fall through to cache → live.
        This is what makes selective replay safe across prompt iteration.

        Replay is a development-only feature. With `enabled=False` (or
        `THRIFTAI_DISABLED=1`) this raises — disabled sessions don't write
        traces, so there's nothing to replay from.

        Args:
            trace_id: ID of a previously-recorded trace to replay from.
            live: Agent names that should re-execute live. Omit or pass an
                empty list to replay everything from the trace (a pure
                no-cost re-run).

        Returns:
            A `ReplayContext` to use as a context manager.

        Raises:
            RuntimeError: If the session is disabled.
            FileNotFoundError: If `trace_id` doesn't exist in the trace store.
            ValueError: If the trace file is malformed.

        Example:
            ```python
            with session.replay(trace_id="run_043", live=["writer"]) as run:
                out = run.completion(messages=[...], model="...")
                print(run.cost_report.summary())
            ```
        """
        if not self.enabled:
            raise RuntimeError(
                "ThriftAI replay is a development-only feature and is "
                "disabled in this session. Set Session(enabled=True) or "
                "unset THRIFTAI_DISABLED to use replay."
            )
        assert self.trace_store is not None  # narrow for type checkers
        replay_trace = self.trace_store.load(trace_id)
        return ReplayContext(
            session=self,
            replay_trace=replay_trace,
            live_agents=list(live or []),
            new_trace_id=_new_trace_id(),
        )


class _BaseRun:
    """Shared plumbing for RunContext and ReplayContext."""

    def __init__(self, session: Session, trace_id: str):
        self.session = session
        self.trace = Trace(trace_id=trace_id)
        self.trace_id = trace_id
        self.cost_report = CostReport()
        self._sequence = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Skip trace recording when disabled — there's no replay path
        # so a trace would just bloat disk for no benefit.
        if self.session.enabled and self.session.trace_store is not None:
            try:
                self.trace.total_cost_usd = self.cost_report.total_cost
                self.session.trace_store.record(self.trace)
            except Exception as e:  # pragma: no cover — defensive
                log.warning("failed to record trace %s: %s", self.trace_id, e)
        if exc_type is None:
            log.info("\n%s", self.cost_report.summary())
        return False

    def _record(
        self,
        agent_name: str,
        messages: list[dict],
        result,
    ) -> None:
        entry = TraceEntry(
            sequence=self._sequence,
            agent_name=agent_name,
            model=result.model,
            messages_hash=compute_prompt_hash(messages) + compute_content_hash(messages),
            response_text=result.response_text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )
        self.trace.entries.append(entry)
        self._sequence += 1

        self.cost_report.entries.append(
            AgentCostEntry(
                agent_name=agent_name,
                resolution=result.resolution.value,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                actual_cost_usd=result.cost_usd,
                would_have_cost_usd=result.cached_cost_usd,
                embedding_cost_usd=result.embedding_cost_usd,
            )
        )


class RunContext(_BaseRun):
    """Context manager returned by [`Session.run`][thriftai.session.Session.run].

    Exposes [`completion()`][thriftai.session.RunContext.completion] for the
    decorated agent body to call, and `cost_report` for the per-agent spend
    summary once the run completes.

    Attributes:
        trace_id: The trace ID assigned to this run.
        cost_report: A [`CostReport`][thriftai.cost.CostReport] populated as
            the run executes.
    """

    def completion(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        """Route an LLM call through the broker.

        Resolution order: exact cache → semantic cache (if enabled) → live.
        The call is attributed to whichever `@agent` is currently executing
        on the thread; if none, it's recorded under the name `"anonymous"`.

        Args:
            messages: OpenAI-style chat messages.
            model: LiteLLM-compatible model identifier
                (e.g. `"anthropic/claude-sonnet-4-20250514"`).
            **kwargs: Forwarded to LiteLLM (temperature, max_tokens, etc.).

        Returns:
            The model response text.
        """
        agent_name = get_current_agent() or "anonymous"
        result = self.session.broker.route(
            messages=messages,
            model=model,
            agent_name=agent_name,
            **kwargs,
        )
        self._record(agent_name, messages, result)
        return result.response_text


class ReplayContext(_BaseRun):
    """Context manager returned by [`Session.replay`][thriftai.session.Session.replay].

    Like [`RunContext`][thriftai.session.RunContext], but `completion()`
    checks the replay trace first and tracks downstream invalidation when a
    live agent diverges from the recorded output.

    Attributes:
        trace_id: The *new* trace ID assigned to this replay (not the
            source trace's ID).
        replay_trace: The loaded source trace being replayed from.
        live_agents: Names of agents that should re-execute live.
        invalidated_agents: Agents whose dependencies have changed during
            this replay and so will skip the replay path.
        cost_report: A [`CostReport`][thriftai.cost.CostReport] tracking
            real spend plus what each replayed call would have cost live.
    """

    def __init__(
        self,
        session: Session,
        replay_trace: Trace,
        live_agents: list[str],
        new_trace_id: str,
    ):
        super().__init__(session=session, trace_id=new_trace_id)
        self.replay_trace = replay_trace
        self.live_agents = live_agents
        self.invalidated_agents: set[str] = set()

    def completion(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        """Route an LLM call through the broker, with replay support.

        Resolution order:

        1. If the current agent is **not** in `live_agents` and **not** in
           `invalidated_agents`, the recorded response is returned from the
           trace (`replay` resolution, $0 cost).
        2. Otherwise, falls through to exact cache → semantic cache → live.

        After step 2 completes, if the produced text differs from the
        recorded output, every transitive dependent of this agent is added
        to `invalidated_agents` so they cannot serve from the now-stale
        trace.

        Args:
            messages: OpenAI-style chat messages.
            model: LiteLLM-compatible model identifier.
            **kwargs: Forwarded to LiteLLM (temperature, max_tokens, etc.).

        Returns:
            The response text — from trace, cache, or live, depending on
            resolution.
        """
        agent_name = get_current_agent() or "anonymous"

        result = self.session.broker.route(
            messages=messages,
            model=model,
            agent_name=agent_name,
            replay_trace=self.replay_trace,
            live_agents=self.live_agents,
            invalidated_agents=self.invalidated_agents,
            **kwargs,
        )

        # Downstream invalidation: if this agent went live (or cache) and its
        # output differs from what the trace recorded, every transitive
        # dependent must skip the replay path on subsequent calls.
        if result.resolution is not CallResolution.REPLAY:
            traced = self.session.trace_store.get_agent_output(
                self.replay_trace, agent_name
            )
            if traced is not None and traced.response_text != result.response_text:
                for dep in get_dependents(agent_name):
                    self.invalidated_agents.add(dep)
                log.debug(
                    "broker: output of %s diverged from trace — invalidating %s",
                    agent_name,
                    sorted(self.invalidated_agents),
                )

        self._record(agent_name, messages, result)
        return result.response_text
