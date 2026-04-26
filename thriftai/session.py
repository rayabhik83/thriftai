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
    """Main entry point. Create one per project/pipeline."""

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
        """Start a normal run. All calls go live, responses are cached and traced."""
        return RunContext(session=self, trace_id=trace_id or _new_trace_id())

    def replay(
        self, trace_id: str, live: list[str] | None = None
    ) -> "ReplayContext":
        """Replay a previous run.

        Agents NOT in `live` list are served from trace.
        Agents in `live` list go through cache -> live.
        If a live agent's output differs from the trace, downstream agents
        are invalidated and fall through to cache -> live.

        Replay is a development-only feature. With `enabled=False` (or
        `THRIFTAI_DISABLED=1`) this raises — there are no traces to load.
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
    """Context manager for a normal (live) run."""

    def completion(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        """Route an LLM call through the broker."""
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
    """Context manager for a replay run."""

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
        """Route through broker with replay + downstream invalidation."""
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
