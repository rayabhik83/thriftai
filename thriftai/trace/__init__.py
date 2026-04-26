"""
Trace — record and replay pipeline runs.

A trace captures every LLM call in a run, ordered by execution sequence.
During replay, agents NOT in the live list are served their exact outputs
from the trace (deterministic, not probabilistic).

Storage: JSON files in {cache_dir}/traces/{trace_id}.json

Trace schema:
{
    "trace_id": "run_043",
    "created_at": "2026-04-19T10:30:00Z",
    "entries": [
        {
            "sequence": 0,
            "agent_name": "observer",
            "model": "claude-sonnet-4-20250514",
            "messages_hash": "abc123",
            "response_text": "...",
            "input_tokens": 150,
            "output_tokens": 320,
            "cost_usd": 0.0024
        },
        ...
    ],
    "total_cost_usd": 0.0048
}

Key distinction:
- Trace replay is DETERMINISTIC: exact outputs from the recorded run
- Cache is PROBABILISTIC: approximate match across different runs
- These are separate mechanisms that complement each other
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TraceEntry:
    """A single LLM call recorded in a trace."""
    sequence: int
    agent_name: str
    model: str
    messages_hash: str
    response_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Trace:
    """A complete pipeline run trace."""
    trace_id: str
    entries: list[TraceEntry] = field(default_factory=list)
    total_cost_usd: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TraceStore:
    """Manages trace recording and loading."""

    def __init__(self, cache_dir: Path):
        self.traces_dir = Path(cache_dir) / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, trace_id: str) -> Path:
        return self.traces_dir / f"{trace_id}.json"

    def record(self, trace: Trace) -> None:
        """Save a trace to disk."""
        payload = {
            "trace_id": trace.trace_id,
            "created_at": trace.created_at,
            "entries": [asdict(e) for e in trace.entries],
            "total_cost_usd": trace.total_cost_usd,
        }
        path = self._path(trace.trace_id)
        path.write_text(json.dumps(payload, indent=2))

    def load(self, trace_id: str) -> Trace:
        """Load a trace from disk.

        Raises:
            FileNotFoundError: If the trace file doesn't exist.
            ValueError: If the file exists but isn't valid trace JSON
                (truncated, hand-edited, key missing, etc.). The original
                JSONDecodeError or KeyError is chained for debugging.
        """
        path = self._path(trace_id)
        if not path.exists():
            raise FileNotFoundError(f"Trace not found: {trace_id} at {path}")

        raw = path.read_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Trace file at {path} is not valid JSON (likely truncated "
                f"by an interrupted run, or hand-edited): {e.msg} at line "
                f"{e.lineno} col {e.colno}"
            ) from e

        try:
            entries = [TraceEntry(**e) for e in data.get("entries", [])]
            return Trace(
                trace_id=data["trace_id"],
                entries=entries,
                total_cost_usd=data.get("total_cost_usd", 0.0),
                created_at=data.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),
            )
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Trace file at {path} is missing required fields or has the "
                f"wrong shape: {e}. Delete the file or restore it from a "
                f"known-good copy."
            ) from e

    def list_traces(self) -> list[str]:
        """List all available trace IDs."""
        return sorted(p.stem for p in self.traces_dir.glob("*.json"))

    def get_agent_output(self, trace: Trace, agent_name: str) -> TraceEntry | None:
        """Get the trace entry for a specific agent."""
        for entry in trace.entries:
            if entry.agent_name == agent_name:
                return entry
        return None
