"""
Cost tracking — per-agent, per-session cost accounting.

The key insight: ThriftAI doesn't just report "this run cost $X."
It reports "this run cost $X and SAVED $Y via cache/replay."

The cost-saved metric is the primary value signal for users.

Pricing data is sourced from LiteLLM's model_cost mapping,
which covers Anthropic, OpenAI, Google, and others.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentCostEntry:
    """Cost tracking for a single agent in a single run."""
    agent_name: str
    resolution: str  # "live", "cache_hit", "semantic_hit", "replay"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    actual_cost_usd: float = 0.0
    would_have_cost_usd: float = 0.0  # what it WOULD have cost if live
    embedding_cost_usd: float = 0.0   # cost of semantic cache lookup / store

    @property
    def saved_usd(self) -> float:
        # Embedding cost is real spend — subtract it from savings.
        return self.would_have_cost_usd - self.actual_cost_usd - self.embedding_cost_usd

    @property
    def total_cost_usd(self) -> float:
        return self.actual_cost_usd + self.embedding_cost_usd


@dataclass
class CostReport:
    """Cost report for an entire run."""
    entries: list[AgentCostEntry] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(e.total_cost_usd for e in self.entries)

    @property
    def total_embedding_cost(self) -> float:
        return sum(e.embedding_cost_usd for e in self.entries)

    @property
    def total_saved(self) -> float:
        return sum(e.saved_usd for e in self.entries)

    @property
    def total_would_have_cost(self) -> float:
        return sum(e.would_have_cost_usd for e in self.entries)

    def summary(self) -> str:
        """Human-readable cost summary."""
        lines = [
            "ThriftAI Cost Report",
            f"{'─' * 50}",
        ]
        for e in self.entries:
            tag = f"[{e.resolution}]"
            line = (
                f"  {e.agent_name:<20} {tag:<14} "
                f"${e.total_cost_usd:.4f}  (saved ${e.saved_usd:.4f})"
            )
            if e.embedding_cost_usd > 0:
                line += f"  [embed ${e.embedding_cost_usd:.4f}]"
            lines.append(line)
        lines.append(f"{'─' * 50}")
        lines.append(f"  Total cost:  ${self.total_cost:.4f}")
        if self.total_embedding_cost > 0:
            lines.append(f"  Embeddings:  ${self.total_embedding_cost:.4f}")
        lines.append(f"  Total saved: ${self.total_saved:.4f}")
        if self.total_would_have_cost > 0:
            pct = (self.total_saved / self.total_would_have_cost) * 100
            lines.append(f"  Savings:     {pct:.0f}%")
        return "\n".join(lines)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost in USD for a completion call.
    Uses LiteLLM's cost mapping: litellm.model_cost
    """
    try:
        import litellm
        return litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception:
        return 0.0
