"""
Session factories — one per benchmark condition.

Conditions and what they mean for the broker:

- `baseline`        Session(enabled=False). Broker still wraps the call but
                    every call falls through to LIVE because the cache is
                    a _NoOpCache. This is what we measure ThriftAI against.
- `thriftai_cold`   Session(enabled=True) with a fresh cache_dir. The
                    day-one user experience.
- `thriftai_warm`   Same as cold, but the runner does an unmeasured
                    warmup pass first to populate the cache. The
                    measured pass then runs against a primed cache —
                    the steady-state user experience.

The replay condition (research_analyst only) is constructed differently
and lives next to that workload.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from thriftai import Session


def make_session(condition: str, cache_dir: Path, thriftai_config: dict) -> Session:
    """Build a Session for a single condition.

    `cache_dir` should already be scoped per-(condition, seed) by the caller
    so that no two cells share a cache directory.
    """
    if condition == "baseline":
        return Session(enabled=False)

    if condition in ("thriftai_cold", "thriftai_warm", "thriftai_replay"):
        return Session(
            cache_dir=cache_dir,
            enabled=True,
            embedding_model=thriftai_config.get("embedding_model"),
            semantic_threshold=thriftai_config.get("semantic_threshold", 0.92),
            semantic_min_chars=thriftai_config.get("semantic_min_chars", 100),
            semantic_bucket_size=thriftai_config.get("semantic_bucket_size", 1000),
        )

    raise ValueError(f"Unknown condition: {condition}")


def reset_cache_dir(cache_dir: Path) -> None:
    """Delete and recreate cache_dir. Used to guarantee cold-start isolation."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
