# ThriftAI

**Make multi-agent LLM development cheaper. Cache, replay, and tier — without changing your pipeline code.**

ThriftAI sits between your orchestration layer (LangGraph, CrewAI, AutoGen, or raw Python) and your LLM provider. It intercepts every call to prevent redundant spend — transparently, without requiring you to change your pipeline logic.

!!! info "ThriftAI is *not* an observability tool"
    Tools like MLflow, Langfuse, and Braintrust already do tracing well. ThriftAI solves the problem they don't: **making the next pipeline run cheaper based on what the last run produced.**

## The problem

Developing multi-agent LLM pipelines is expensive because:

- **Redundant calls** — tweaking one agent's prompt re-runs the entire pipeline, paying for all unchanged agents
- **Iteration loops** — prompt engineering is trial-and-error; each experiment is a full API round-trip
- **No selective re-execution** — you can't iterate on agent 3 without re-paying for agents 1 and 2

## How it works

ThriftAI uses a **decision cascade** for every LLM call:

1. **Replay check** → Is this agent being replayed? Serve exact output from trace.
2. **Cache check** → Is there an exact-match (or semantic) hit? Serve cached response.
3. **Live call** → Route to LLM. Record in cache and trace. Track cost.

See [Concepts](concepts.md) for the full model.

## Where to go next

- **[Quick Start](quickstart.md)** — install, run an example pipeline, replay a single agent
- **[Concepts](concepts.md)** — the decision cascade, downstream invalidation, replay vs. cache
- **[Production Guide](production.md)** — when to enable, kill switch, break-even analysis, known gaps
- **[API Reference](api.md)** — full surface, generated from source

## Features at a glance

- **Selective replay** — replay N-1 agents from trace, send 1 live
- **Exact-match cache** — hash-based, scoped per agent + prompt template
- **Semantic cache** (opt-in) — embeddings-based fuzzy match
- **Downstream invalidation** — if a live agent's output changes during replay, dependents auto-invalidate
- **Cost-saved metric** — reports what you saved, not just what you spent
- **Provider-agnostic** — works with any provider via LiteLLM
- **Zero lock-in** — decorator/wrapper pattern; keep your existing pipeline code
