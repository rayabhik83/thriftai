# Quick Start

## Install

```bash
pip install thriftai
```

To enable the semantic cache (optional, fuzzy matching via embeddings):

```bash
pip install 'thriftai[semantic]'
```

## A two-agent pipeline

```python
import thriftai as ta

@ta.agent(name="researcher")
def research(run, topic):
    return run.completion(
        messages=[{"role": "user", "content": f"Research: {topic}"}],
        model="anthropic/claude-sonnet-4-20250514",
    )

@ta.agent(name="writer", depends_on=["researcher"])
def write(run, research):
    return run.completion(
        messages=[{"role": "user", "content": f"Summarize: {research}"}],
        model="anthropic/claude-sonnet-4-20250514",
    )

session = ta.Session()
```

The `@agent` decorator registers each function in a DAG. `depends_on` tells ThriftAI that `writer` consumes `researcher`'s output — used later for [downstream invalidation](concepts.md#downstream-invalidation).

## Run it live

```python
with session.run() as run:
    data = research(run, "AI costs")
    summary = write(run, data)
    print(run.cost_report.summary())
```

Both agents go live. Their outputs and costs are recorded into a trace under `.thriftai/traces/`.

```
ThriftAI Cost Report
──────────────────────────────────────────────────
  researcher           [live]       $0.3600  (saved $0.0000)
  writer               [live]       $0.0700  (saved $0.0000)
──────────────────────────────────────────────────
  Total cost:  $0.4300
  Total saved: $0.0000
```

## Iterate on a single agent

You want to tweak the writer's prompt without re-paying for the researcher. Grab the trace ID and replay:

```python
trace_id = run.trace_id  # e.g. "run_20260516_154523"

with session.replay(trace_id=trace_id, live=["writer"]) as run:
    data = research(run, "AI costs")        # served from trace, $0
    summary = write(run, data)              # goes live
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

The pipeline code didn't change. Only the wrapping context did.

## Disable for production-only paths

When every call is unique (interactive chat), the cache is pure overhead. Turn ThriftAI off without changing pipeline code:

```python
session = ta.Session(enabled=False)
```

Or globally:

```bash
THRIFTAI_DISABLED=1 python my_app.py
```

Cost tracking still works; cache and replay become no-ops. See the [Production Guide](production.md) for when to use each mode.

## Next steps

- Read [Concepts](concepts.md) to understand replay vs. cache, downstream invalidation, and the decision cascade.
- Read the [Production Guide](production.md) before enabling ThriftAI in a deployed service.
- See [API Reference](api.md) for the full `Session`, `agent`, and `CostReport` surface.
