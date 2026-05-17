# Concepts

## The decision cascade

Every `run.completion(...)` call passes through a fixed resolution order:

```
┌─────────────────┐
│  Replay check   │  Is the current agent being replayed?
└────────┬────────┘  If yes → return the trace entry. Done.
         │ no
         ▼
┌─────────────────┐
│  Exact cache    │  Is there a hash match in the SQLite cache?
└────────┬────────┘  If yes → return cached response. Done.
         │ no
         ▼
┌─────────────────┐
│ Semantic cache  │  (Optional) Is there a cosine-similar entry?
└────────┬────────┘  If yes → return matched response. Done.
         │ no
         ▼
┌─────────────────┐
│   Live call     │  Send to LiteLLM. Record into cache + trace.
└─────────────────┘  Return response.
```

Each layer is independent. You can disable any of them:

- Disable cache + replay: `Session(enabled=False)`
- Disable semantic only: omit `embedding_model=` (this is the default)
- Disable replay only: just don't call `Session.replay()`

## Trace vs. cache

Both store past responses, but they answer different questions.

| | Trace | Cache |
|---|---|---|
| **Scope** | One specific run | All runs, forever |
| **Key** | Sequence of agent calls in that run | `(agent, prompt_hash, content_hash)` |
| **Matching** | Deterministic — exact replay of recorded steps | Probabilistic — hash or cosine match |
| **Use case** | "Re-run this exact pipeline, swap one agent" | "Skip this call if I've seen its inputs before" |

Trace replay is **deterministic**: the exact text the model produced last time is returned. Cache hits are **probabilistic**: a different run with the same inputs gets the cached response, but a hash mismatch by one character is a miss.

## Replay

`Session.replay(trace_id=..., live=[...])` re-runs a pipeline against a stored trace.

- Agents named in `live=[...]` go through the normal cache → live cascade.
- All other agents are served from the trace: $0 cost, deterministic output.

This is **dev-only**. Replay has no production use case, and calling it with `enabled=False` raises.

### Downstream invalidation

What happens if a `live` agent produces *different* output than what's in the trace? The recorded outputs for its dependents are now stale — they were produced from inputs that no longer match reality.

ThriftAI handles this automatically. After every live (or cache) resolution during a replay, the broker compares the produced text to the recorded trace entry:

- If they match → continue using the trace for downstream agents.
- If they differ → mark every transitive dependent of this agent as invalidated. Those agents skip the replay path on subsequent calls and fall through to cache → live.

This is why `depends_on=[...]` on the `@agent` decorator matters: it's the graph used for invalidation propagation. The runtime does **not** use it to schedule execution; you call agents in whatever order your pipeline code dictates.

## Cache scoping

Cache entries are keyed on `(agent_name, prompt_hash, content_hash)`:

- `agent_name` — from the `@agent` decorator's thread-local
- `prompt_hash` — SHA-256 of the system messages
- `content_hash` — SHA-256 of the non-system messages

Two agents that send identical messages do **not** share cache entries. This is intentional: an `observer` and a `synthesizer` will usually produce different outputs for the same input, so colliding their caches would be a bug.

Renaming an agent invalidates its cache (because `agent_name` is part of the key). Editing a system prompt invalidates its cache. Both are usually what you want.

To manually invalidate after a model upgrade or data refresh:

```python
session.cache.invalidate_agent("researcher")
```

There is no TTL. See the [Production Guide](production.md#known-gaps) for the workaround.

## Cost tracking

Every `completion()` adds an [`AgentCostEntry`][thriftai.cost.AgentCostEntry] to the run's [`CostReport`][thriftai.cost.CostReport]. Each entry records:

- What the call actually cost (`actual_cost_usd`) — usually $0 for cache/replay
- What it *would* have cost live (`would_have_cost_usd`)
- Any embedding overhead from semantic-cache lookups (`embedding_cost_usd`)

The `saved_usd` field nets out the embedding overhead, so semantic cache that costs more than it saves shows as negative. See the [Production Guide](production.md#cost-crossover) for break-even thresholds per model.
