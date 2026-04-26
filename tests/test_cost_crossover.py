"""
Cost crossover tests: when does ThriftAI's semantic cache *cost* more than
the LLM call it tries to avoid?

Per-query expected savings:

    E[savings] = p × C_LLM − C_embed

Break-even hit rate:

    p_breakeven = C_embed / C_LLM

Above `p_breakeven`, ThriftAI saves money. Below, it loses. The
`min_query_chars` knob exists to short-circuit the disastrous regime where
`C_embed ≈ C_LLM` (tiny prompts to cheap models).

These tests are **hermetic** — they use synthetic per-token rates so the
results don't drift with LiteLLM pricing-table updates. The numbers match
the table in `STRESS_REPORT.md` at the repo root.
"""

from __future__ import annotations

import pytest


# Synthetic rates ($ per token). Roughly matches commercial pricing as of
# 2026-Q2 — adjust here if a follow-up wants to track real numbers.
PRICING = {
    "claude-sonnet-4":   {"in_per_token": 3.0e-6,  "out_per_token": 15.0e-6},
    "claude-haiku-4-5":  {"in_per_token": 0.25e-6, "out_per_token": 1.25e-6},
    "gpt-4o-mini":       {"in_per_token": 0.15e-6, "out_per_token": 0.60e-6},
    "gpt-4o":            {"in_per_token": 2.5e-6,  "out_per_token": 10.0e-6},
}

# text-embedding-3-small ≈ $0.02 / 1M tokens. ~1 token per 4 chars in English,
# so a typical 500-char query is ~125 tokens × $0.02e-6 = $2.5e-6.
EMBED_PER_TOKEN = 0.02e-6


def llm_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING[model]
    return p["in_per_token"] * in_tok + p["out_per_token"] * out_tok


def embed_cost(in_tok: int) -> float:
    return EMBED_PER_TOKEN * in_tok


def breakeven(model: str, in_tok: int, out_tok: int) -> float:
    return embed_cost(in_tok) / llm_cost(model, in_tok, out_tok)


# ---------------------------------------------------------------------------
# Per-model break-even points (table in STRESS_REPORT.md)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,in_tok,out_tok,upper_bound",
    [
        # Frontier-tier models: caching almost always wins.
        ("claude-sonnet-4", 500, 200, 0.01),
        ("gpt-4o",          500, 200, 0.01),
        # Mid-tier: still profitable above ~5% hit rate.
        ("claude-haiku-4-5", 500, 200, 0.06),
        ("gpt-4o-mini",      500, 200, 0.06),
    ],
)
def test_breakeven_below_upper_bound(model, in_tok, out_tok, upper_bound):
    p = breakeven(model, in_tok, out_tok)
    assert p < upper_bound, (
        f"{model} typical-query break-even is {p:.4f}, expected <{upper_bound}"
    )


# ---------------------------------------------------------------------------
# Tiny-query regime: where ThriftAI hurts without min_query_chars
# ---------------------------------------------------------------------------


def test_breakeven_higher_for_cheap_than_frontier_models():
    """Across model tiers, cheap models require a higher cache hit rate
    to break even. This is the real-world signal that semantic caching
    pays off most for slow/expensive models."""
    in_tok, out_tok = 500, 200
    p_frontier = breakeven("claude-sonnet-4", in_tok, out_tok)
    p_cheap    = breakeven("claude-haiku-4-5", in_tok, out_tok)
    # Frontier models pay back almost immediately; cheap models need a
    # meaningfully higher hit rate. The ratio reflects the gap in per-call cost.
    assert p_cheap > p_frontier * 5, (
        f"cheap-vs-frontier breakeven ratio {p_cheap/p_frontier:.1f}x "
        f"is unexpectedly low; pricing assumptions may have drifted"
    )


# Note on `min_query_chars`: pure-math break-even doesn't strictly require
# the skip — input and output token costs both scale linearly, so the ratio
# C_embed/C_LLM stays roughly constant across query sizes. The real
# justifications for the default are:
#   1. **Latency.** An embedding round-trip adds 50–200 ms even on a hit.
#      For sub-100-char queries, that's often slower than the LLM call.
#   2. **Cardinality.** Short queries (function calls, tool names, status
#      codes) are usually unique per call; cache hit rate ≈ 0, so every
#      embedding is wasted.
#   3. **Embedding quality.** Short text produces noisy embeddings;
#      similarity scores are unreliable below ~10 tokens.
# The cost-control test for that knob lives in tests/test_cost_controls.py
# (it asserts behavioral skipping, not arithmetic break-even).


# ---------------------------------------------------------------------------
# Cost formula sanity
# ---------------------------------------------------------------------------


def test_savings_formula_zero_at_breakeven():
    """At the break-even hit rate, expected per-query savings is zero."""
    in_tok, out_tok = 500, 200
    model = "claude-haiku-4-5"
    p = breakeven(model, in_tok, out_tok)
    expected_savings = p * llm_cost(model, in_tok, out_tok) - embed_cost(in_tok)
    assert abs(expected_savings) < 1e-12


def test_savings_positive_above_breakeven():
    in_tok, out_tok = 500, 200
    model = "claude-haiku-4-5"
    p = breakeven(model, in_tok, out_tok) + 0.10  # 10pp above break-even
    expected_savings = p * llm_cost(model, in_tok, out_tok) - embed_cost(in_tok)
    assert expected_savings > 0


def test_savings_negative_below_breakeven():
    in_tok, out_tok = 20, 10
    model = "gpt-4o-mini"
    p = breakeven(model, in_tok, out_tok) / 2  # half the break-even
    expected_savings = p * llm_cost(model, in_tok, out_tok) - embed_cost(in_tok)
    assert expected_savings < 0
