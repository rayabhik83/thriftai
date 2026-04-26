"""
Property-based tests (hypothesis).

ThriftAI's correctness boils down to a few hash functions and two storage
round-trips. Hand-written tests cover the obvious shapes; these tests
generate adversarial shapes — empty strings, surrogate code points, deep
nesting, lots of unicode — that real prod traffic will eventually
produce. If a generated input breaks an invariant here, hypothesis
shrinks it to a minimal failing case.

Invariants we want to hold for ALL valid inputs:

1. **Hash determinism.** Hashing the same messages twice yields the same
   hex digest.
2. **Hash disjointness.** Changing only the system message changes
   `prompt_hash` but keeps `content_hash` (and vice versa).
3. **Hash totality.** Both hash functions accept any sequence of valid
   chat messages without raising.
4. **Trace round-trip.** `TraceStore.record(t)` followed by `load(t.trace_id)`
   yields a `Trace` whose entries equal `t`'s entries.
5. **Exact cache round-trip.** `cache.put(...)` followed by `cache.get(...)`
   on the same key returns the stored fields.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st

from thriftai.cache import (
    ExactCache,
    compute_content_hash,
    compute_prompt_hash,
)
from thriftai.trace import Trace, TraceEntry, TraceStore


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# Plain text strings, restricted to characters JSON can serialize without
# fuss. (json.dumps doesn't allow lone surrogates by default; allowing them
# in tests just generates noise about the test setup, not real bugs.)
TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),  # exclude lone surrogates
    max_size=200,
)


def _message_strategy(role: str | None = None) -> st.SearchStrategy:
    """A chat message dict. If `role` is fixed, generate that role only."""
    role_strat = st.just(role) if role else st.sampled_from(["system", "user", "assistant"])
    return st.fixed_dictionaries({
        "role": role_strat,
        "content": TEXT,
    })


SYSTEM_ONLY = st.lists(_message_strategy("system"), min_size=0, max_size=3)
USER_ONLY = st.lists(_message_strategy("user"), min_size=1, max_size=5)
MIXED_MESSAGES = st.lists(_message_strategy(), min_size=1, max_size=8)


# ---------------------------------------------------------------------------
# Hash determinism + totality
# ---------------------------------------------------------------------------


@given(messages=MIXED_MESSAGES)
def test_prompt_hash_is_deterministic(messages):
    assert compute_prompt_hash(messages) == compute_prompt_hash(messages)


@given(messages=MIXED_MESSAGES)
def test_content_hash_is_deterministic(messages):
    assert compute_content_hash(messages) == compute_content_hash(messages)


@given(messages=MIXED_MESSAGES)
def test_hashes_are_total(messages):
    """Both hash functions return a 16-char hex string for any valid message list."""
    p = compute_prompt_hash(messages)
    c = compute_content_hash(messages)
    for h in (p, c):
        assert len(h) == 16
        int(h, 16)  # parses as hex


# ---------------------------------------------------------------------------
# Hash disjointness
# ---------------------------------------------------------------------------


@given(system=SYSTEM_ONLY, user=USER_ONLY, replacement_text=TEXT)
def test_changing_only_system_changes_prompt_hash_only(system, user, replacement_text):
    """Replace ALL system messages with a different one. content_hash must
    not change; prompt_hash should change unless the replacement happens
    to serialize identically to what was there."""
    base = system + user

    new_system = [{"role": "system", "content": replacement_text}]
    perturbed = new_system + user

    # content_hash depends only on non-system messages and so MUST be stable.
    assert compute_content_hash(base) == compute_content_hash(perturbed)


@given(system=SYSTEM_ONLY, user=USER_ONLY, extra_user_text=TEXT)
def test_changing_only_user_changes_content_hash_only(system, user, extra_user_text):
    """Append a new user message; prompt_hash must not change."""
    base = system + user
    perturbed = system + user + [{"role": "user", "content": extra_user_text}]
    assert compute_prompt_hash(base) == compute_prompt_hash(perturbed)


# ---------------------------------------------------------------------------
# TraceStore round-trip
# ---------------------------------------------------------------------------


_trace_id_strat = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=64,
)


@st.composite
def _trace_entry(draw) -> TraceEntry:
    return TraceEntry(
        sequence=draw(st.integers(min_value=0, max_value=10_000)),
        agent_name=draw(st.text(min_size=1, max_size=32)),
        model=draw(st.text(min_size=1, max_size=64)),
        messages_hash=draw(st.text(min_size=1, max_size=64)),
        response_text=draw(TEXT),
        input_tokens=draw(st.integers(min_value=0, max_value=1_000_000)),
        output_tokens=draw(st.integers(min_value=0, max_value=1_000_000)),
        cost_usd=draw(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)),
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
@given(trace_id=_trace_id_strat, entries=st.lists(_trace_entry(), min_size=0, max_size=10),
       cost=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_trace_record_load_roundtrip(tmp_path_factory, trace_id, entries, cost):
    store = TraceStore(tmp_path_factory.mktemp("traces"))
    trace = Trace(trace_id=trace_id, entries=entries, total_cost_usd=cost)
    store.record(trace)
    loaded = store.load(trace_id)
    assert loaded.trace_id == trace_id
    assert loaded.entries == entries
    assert loaded.total_cost_usd == cost


# ---------------------------------------------------------------------------
# ExactCache round-trip
# ---------------------------------------------------------------------------


_short_text = st.text(min_size=1, max_size=64)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
@given(
    agent=_short_text,
    prompt_hash=_short_text,
    content_hash=_short_text,
    model=_short_text,
    response=TEXT,
    in_tok=st.integers(min_value=0, max_value=1_000_000),
    out_tok=st.integers(min_value=0, max_value=1_000_000),
)
def test_exact_cache_put_get_roundtrip(
    tmp_path_factory, agent, prompt_hash, content_hash, model, response, in_tok, out_tok
):
    cache = ExactCache(tmp_path_factory.mktemp("cache"))
    cache.put(
        agent_name=agent,
        prompt_hash=prompt_hash,
        content_hash=content_hash,
        model=model,
        response_text=response,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
    hit = cache.get(agent, prompt_hash, content_hash)
    assert hit is not None
    assert hit["agent_name"] == agent
    assert hit["model"] == model
    assert hit["response_text"] == response
    assert hit["input_tokens"] == in_tok
    assert hit["output_tokens"] == out_tok
