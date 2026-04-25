# ThriftAI

**Make multi-agent LLM development cheaper. Cache, replay, and tier — without changing your pipeline code.**

ThriftAI sits between your orchestration layer (LangGraph, CrewAI, AutoGen, or raw Python) and your LLM provider. It intercepts every call to prevent redundant spend — transparently, without requiring you to change your pipeline logic.

ThriftAI is **not** an observability tool, a tracing platform, or an LLM gateway. Tools like MLflow, Langfuse, and Braintrust already do those jobs well. ThriftAI solves the problem they don't: **making the next pipeline run cheaper based on what the last run produced.**

## The Problem

Developing multi-agent LLM pipelines is expensive because:

- **Redundant calls** — tweaking one agent's prompt re-runs the entire pipeline, paying for all unchanged agents
- **Iteration loops** — prompt engineering is trial-and-error; each experiment is a full API round-trip
- **No selective re-execution** — you can't iterate on agent 3 without re-paying for agents 1 and 2

## Quick Start

```bash
pip install thriftai
```

```python
import thriftai as ta

@ta.agent(name="researcher")
def research(session, topic):
    return session.completion(
        messages=[{"role": "user", "content": f"Research: {topic}"}],
        model="anthropic/claude-sonnet-4-20250514",
    )

@ta.agent(name="writer", depends_on=["researcher"])
def write(session, research):
    return session.completion(
        messages=[{"role": "user", "content": f"Summarize: {research}"}],
        model="anthropic/claude-sonnet-4-20250514",
    )

session = ta.Session()

# Run 1: both agents go live — $0.43
with session.run() as run:
    data = research(run, "AI costs")
    summary = write(run, data)

# Run 2: only writer goes live, researcher replays from trace — $0.07
with session.replay(trace_id=run.trace_id, live=["writer"]) as run:
    data = research(run, "AI costs")
    summary = write(run, data)
    print(run.cost_report.summary())
```

```
ThriftAI Cost Report
──────────────────────────────────────────────────
  researcher           [replay]     $0.0000  (saved $0.3600)
  writer               [live]       $0.0700  (saved $0.0000)
──────────────────────────────────────────────────
  Total cost:  $0.0700
  Total saved: $0.3600
  Savings:     84%
```

## How It Works

ThriftAI uses a **decision cascade** for every LLM call:

1. **Replay check** → Is this agent being replayed? Serve exact output from trace.
2. **Cache check** → Is there an exact-match hit? Serve cached response.
3. **Live call** → Route to LLM. Record in cache and trace. Track cost.

## Features

- **Selective replay**: Replay N-1 agents from trace, send 1 live
- **Exact-match cache**: Hash-based, scoped per agent + prompt template
- **Downstream invalidation**: If a live agent's output changes during replay, dependents auto-invalidate
- **Cost-saved metric**: Reports what you saved, not just what you spent
- **Provider-agnostic**: Works with any provider via LiteLLM (Anthropic, OpenAI, Google, etc.)
- **Zero lock-in**: Decorator/wrapper pattern — keep your existing pipeline code

## Should I use ThriftAI in production?

Short answer: **probably yes, with caveats.** Use it where inputs recur, disable it where every call is unique.

### When it pays off

- **Batch / scheduled agent pipelines.** Nightly summarization, weekly research bots, daily reports — same inputs recur. Cache hit rates often >50%.
- **Eval and benchmark loops.** Re-running the same prompts across models. Hit rate ≈100% after the first pass.
- **RAG with long-tail recurrence.** Many users asking the same questions of your docs.

### When it's a net loss

- **Interactive user-facing chat** where every prompt is unique. Cache hit rate ≈0; you pay storage + lookup overhead for nothing.
- **Cheap models with cheap embeddings.** Below a per-call cost threshold, semantic caching costs more than it saves. A `STRESS_REPORT.md` with the per-model break-even table is on the way.
- **Hard p99 SLAs.** SQLite writes add 1–2 ms; measure first.

### Replay is dev-only

`Session.replay()` exists for prompt iteration during development. It has no production use; calling it with `enabled=False` raises.

### Kill switch

Two equivalent ways to disable cache + replay (cost tracking stays on):

```python
session = Session(enabled=False)             # per-session
```

```bash
THRIFTAI_DISABLED=1 python my_app.py         # global, wins over the kwarg
```

When disabled, `Session` is a thin pass-through to LiteLLM. No filesystem writes, no embedding calls, no traces. `CostReport` still summarizes per-agent spend.

### Open production gaps

Be transparent about what's not solved yet:

- **No TTL** on cached responses. Invalidate manually with `cache.invalidate_agent(name)` after a model upgrade or data refresh.
- **Single-instance cache.** Each replica has its own SQLite. Use a shared volume, or wait for the planned Redis backend.
- **Response text stored unencrypted.** If agent inputs are sensitive, encrypt the cache directory at rest or run with `enabled=False` until the planned PII-redaction layer lands.

## License

MIT
