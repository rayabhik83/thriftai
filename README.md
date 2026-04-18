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

## License

MIT
