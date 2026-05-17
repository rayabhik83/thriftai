"""
Agent — decorator that registers a function as a named agent in the DAG.

Usage:
    @ta.agent(name="observer", depends_on=[])
    def observe(session, input_data):
        return session.completion(messages=[...], model="claude-sonnet-4-20250514")

    @ta.agent(name="hypothesizer", depends_on=["observer"])
    def hypothesize(session, observation):
        return session.completion(messages=[...], model="claude-sonnet-4-20250514")

Design notes:
- The decorator registers the agent in a global DAG registry
- When the decorated function executes, it sets a thread-local _current_agent
  so the broker knows which agent is making the LLM call
- depends_on is used for downstream invalidation during replay:
  if agent A's output changes, all agents that depend on A are invalidated
- The prompt_template_hash is computed lazily on first completion() call
  by hashing the system message content. This is used for cache keying.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

# Thread-local to track which agent is currently executing
_current_agent: threading.local = threading.local()


@dataclass
class AgentMeta:
    """Metadata for a registered agent."""
    name: str
    depends_on: list[str] = field(default_factory=list)
    prompt_template_hash: str | None = None  # computed lazily


# Global registry: agent_name -> AgentMeta
_agent_registry: dict[str, AgentMeta] = {}


def agent(name: str, depends_on: list[str] | None = None) -> Callable:
    """Register a function as a named agent in the pipeline DAG.

    The decorated function's body should call `run.completion(...)` (or
    `replay_ctx.completion(...)`) — those calls are attributed to this
    agent's `name` via a thread-local, so the broker scopes cache and trace
    operations per-agent.

    The `depends_on` list drives **downstream invalidation** during replay:
    if this agent's output changes mid-replay, every agent that lists this
    one in its `depends_on` (directly or transitively) is forced off the
    replay path for the rest of the run.

    Decorated functions are safe to nest. When agent A calls agent B which
    calls agent A, the inner `agent` correctly restores the outer agent on
    unwind.

    Args:
        name: Unique name for the agent in the registry. Used as part of
            the cache key, so renaming an agent invalidates its prior cache
            entries (which is usually what you want).
        depends_on: Names of agents this one consumes the output of. Used
            only for replay invalidation; the runtime does not enforce
            execution order.

    Returns:
        A decorator that wraps the function and registers it.

    Example:
        ```python
        @ta.agent(name="researcher")
        def research(run, topic):
            return run.completion(
                messages=[{"role": "user", "content": f"Research: {topic}"}],
                model="anthropic/claude-sonnet-4-20250514",
            )

        @ta.agent(name="writer", depends_on=["researcher"])
        def write(run, research):
            return run.completion(
                messages=[{"role": "user", "content": f"Summarize: {research}"}],
                model="anthropic/claude-sonnet-4-20250514",
            )
        ```
    """
    def decorator(fn: Callable) -> Callable:
        meta = AgentMeta(name=name, depends_on=depends_on or [])
        _agent_registry[name] = meta

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Save the previous value so nested @agent calls (A → B → A)
            # restore the outer agent on unwind instead of clobbering it to None.
            previous = getattr(_current_agent, "name", None)
            _current_agent.name = name
            try:
                return fn(*args, **kwargs)
            finally:
                _current_agent.name = previous

        wrapper._thriftai_meta = meta  # type: ignore
        return wrapper

    return decorator


def get_current_agent() -> str | None:
    """Get the name of the currently executing agent."""
    return getattr(_current_agent, "name", None)


def get_agent_registry() -> dict[str, AgentMeta]:
    """Get the global agent registry."""
    return _agent_registry


def get_dependents(agent_name: str) -> list[str]:
    """Get all agents that depend on the given agent (direct + transitive).

    Returns names in BFS order, without duplicates. The root agent is not
    included in the result.
    """
    result: list[str] = []
    seen: set[str] = {agent_name}
    frontier = [agent_name]
    while frontier:
        current = frontier.pop(0)
        for name, meta in _agent_registry.items():
            if current in meta.depends_on and name not in seen:
                seen.add(name)
                result.append(name)
                frontier.append(name)
    return result
