# ThriftAI Benchmark Results

> Generated 2026-05-20 23:49 UTC from 45 calls across 3 run(s).
> Pricing snapshot: pulled 2026-05-19 — [source](https://www.anthropic.com/pricing#anthropic-api).

## Headline

| Workload | Condition | Model | $/task paid (mean ± std) | $/task saved (mean ± std) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|
| support_triage | baseline | claude-haiku-4-5 | 0.0003 ± 0.0000 $ | 0.0000 ± 0.0000 $ | 882 | 2209 |
| support_triage | thriftai_cold | claude-haiku-4-5 | 0.0003 ± 0.0000 $ | 0.0000 ± 0.0000 $ | 908 | 1766 |
| support_triage | thriftai_warm | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0003 ± 0.0000 $ | 1 | 1 |

## Call resolution breakdown

Counts of brokered-call outcomes per cell. Cache vs replay vs live
tells you which mechanism is doing the work.

| Workload | Condition | Model | live | cache_hit | semantic_hit | replay |
|---|---|---|---:|---:|---:|---:|
| support_triage | baseline | claude-haiku-4-5 | 15 | 0 | 0 | 0 |
| support_triage | thriftai_cold | claude-haiku-4-5 | 15 | 0 | 0 | 0 |
| support_triage | thriftai_warm | claude-haiku-4-5 | 0 | 15 | 0 | 0 |

## Per-workload deep dives

_filled in once workloads land._

## Methodology

See `benchmarks/README.md` and `benchmarks/PLAN.md`.

## Raw data

Per-call records under `benchmarks/results/raw/<run_id>/calls.jsonl`.
Every dollar figure above is derived from raw token counts in those
files multiplied by `benchmarks/pricing.yaml`; see `make rederive`
for the verification script.
