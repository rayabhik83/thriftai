# Production Guide

Short answer: **probably yes, with caveats**. Use ThriftAI where inputs recur, disable it where every call is unique.

## When it pays off

- **Batch / scheduled agent pipelines.** Nightly summarization, weekly research bots, daily reports — same inputs recur. Cache hit rates often >50%.
- **Eval and benchmark loops.** Re-running the same prompts across models. Hit rate ≈100% after the first pass.
- **RAG with long-tail recurrence.** Many users asking the same questions of your docs.

## When it's a net loss

- **Interactive user-facing chat** where every prompt is unique. Cache hit rate ≈0; you pay storage + lookup overhead for nothing.
- **Cheap models with cheap embeddings.** Below a per-call cost threshold, semantic caching costs more than it saves. See [Cost crossover](#cost-crossover) below.
- **Hard p99 SLAs.** SQLite writes add 1–2 ms; measure first.

## Kill switch

Two equivalent ways to disable cache + replay (cost tracking stays on):

```python
session = Session(enabled=False)             # per-session
```

```bash
THRIFTAI_DISABLED=1 python my_app.py         # global, wins over the kwarg
```

When disabled, `Session` is a thin pass-through to LiteLLM. No filesystem writes, no embedding calls, no traces. [`CostReport`][thriftai.cost.CostReport] still summarizes per-agent spend.

!!! warning "Replay is dev-only"
    `Session.replay()` exists for prompt iteration during development. It has no production use; calling it with `enabled=False` raises.

## Cost crossover

Per-query expected savings from the semantic cache:

```
E[savings] = p × C_LLM − C_embed
```

where `p` is the empirical cache hit rate. Break-even hit rate:

```
p_breakeven = C_embed / C_LLM
```

Above `p_breakeven`, ThriftAI saves money. Below, it costs money. Assuming `text-embedding-3-small` at \$0.02 per 1M tokens:

| Model | Input | Output | C_LLM | C_embed | **p_breakeven** |
|---|---:|---:|---:|---:|---:|
| `claude-sonnet-4` | 500 | 200 | \$0.0045 | \$0.00001 | **0.22%** |
| `gpt-4o` | 500 | 200 | \$0.0033 | \$0.00001 | **0.31%** |
| `claude-haiku-4-5` | 500 | 200 | \$0.00038 | \$0.00001 | **2.67%** |
| `gpt-4o-mini` | 500 | 200 | \$0.00020 | \$0.00001 | **5.13%** |

A 5% break-even on `gpt-4o-mini` is not a problem in practice — most realistic agent workloads cache at 30–80% hit rates. Use the break-even as a *floor* sanity check, not a target.

The numbers above are regenerated on every PR by `tests/test_cost_crossover.py`. The exact-match cache has no embedding cost and is profitable as long as you ever get a hit.

## Tuning knobs (all on `Session`)

| Param | Default | What it does |
|---|---|---|
| `enabled` | `True` | Master kill switch. Honors `THRIFTAI_DISABLED=1`. |
| `embedding_model` | `None` | Set to enable semantic cache. `None` keeps it off. |
| `semantic_threshold` | `0.92` | Cosine similarity floor for a hit. Higher = fewer wrong-hits, more false misses. |
| `semantic_min_chars` | `100` | Skip embedding for queries shorter than this. Saves latency on tiny inputs where caching has near-zero value. |
| `semantic_bucket_size` | `1000` | Cap entries per bucket. FIFO eviction. Keeps similarity scan latency bounded. |

## Wrong-hit risk

At the default threshold of `0.92`, the semantic cache reliably catches paraphrases but **does not** distinguish:

- **Negation** (`is X?` vs `is X not?`) — embeddings score these as near-identical.
- **Numeric drift** (`Compute 2+2` vs `Compute 2+3`) — lexical similarity dominates.
- **Entity swaps** (`CEO of Apple` vs `CEO of Google`) — single proper-noun changes get diluted.
- **Date drift** (`Q1 2024` vs `Q1 2025`) — same risk class.

If your workload includes financial Q&A, scientific computation, or real-time entity lookup, either raise the threshold to `0.95+` or **disable semantic caching** for the affected agents (the exact cache stays on; it doesn't have this failure mode).

Per-category measurements live in `tests/adversarial_report.md`, regenerated against a real embedding model when `THRIFTAI_LIVE_TEST=1` is set.

## Known gaps

Tracked but not yet solved:

- **No TTL** on cached responses. Invalidate manually with `session.cache.invalidate_agent(name)` after a model upgrade or data refresh.
- **Single-instance SQLite.** Each replica has its own cache file. Use a shared volume, or wait for the planned Redis backend.
- **Response text stored unencrypted.** If agent inputs are sensitive, encrypt the cache directory at rest or run with `enabled=False` until the planned PII-redaction layer lands.
- **Linear scan within a bucket** — capped via `bucket_size` (default 1000). Above ~10k entries per bucket the scan starts to bite; the scale tests in `tests/test_scale.py` (gated on `pytest -m slow`) assert p95 lookup <100 ms at 10k entries.

## Operational checklist

Before enabling ThriftAI in a deployed service:

- [ ] Measure your input cardinality. If every request is unique, set `enabled=False`.
- [ ] Pick an `embedding_model` — or leave it off and use exact-match cache only.
- [ ] Confirm your model is above the break-even hit rate for your workload.
- [ ] Check that your agent inputs don't contain PII (or accept that they'll land on disk in plain text).
- [ ] Mount `.thriftai/` (or your `cache_dir=`) on persistent storage. Container restarts otherwise wipe the cache.
- [ ] Add `session.cache.invalidate_agent(...)` calls to any post-deploy hook that swaps models.
