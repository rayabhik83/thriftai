# ThriftAI Benchmark Results

> **Status: partial.** 1/4 workloads complete: support_triage. Pending: research_analyst, code_review, humaneval.
>
> Generated 2026-05-21 01:00 UTC from 360 calls across 6 run(s).
> Pricing snapshot: pulled 2026-05-19 — [source](https://www.anthropic.com/pricing#anthropic-api).

## Headline

| Workload | Condition | Model | $/task paid (mean ± std) | $/task saved (mean ± std) | Quality (1-5, mean ± std) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|---|
| support_triage | baseline | claude-haiku-4-5 | 0.0003 ± 0.0000 $ | 0.0000 ± 0.0000 $ | 4.47 ± 0.34 | 772 | 2202 |
| support_triage | thriftai_cold | claude-haiku-4-5 | 0.0003 ± 0.0000 $ | 0.0000 ± 0.0000 $ | 4.47 ± 0.39 | 762 | 1972 |
| support_triage | thriftai_warm | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0003 ± 0.0000 $ | 4.42 ± 0.35 | 0 | 1 |

## Call resolution breakdown

Counts of brokered-call outcomes per cell. Cache vs replay vs live
tells you which mechanism is doing the work.

| Workload | Condition | Model | live | cache_hit | semantic_hit | replay |
|---|---|---|---:|---:|---:|---:|
| support_triage | baseline | claude-haiku-4-5 | 120 | 0 | 0 | 0 |
| support_triage | thriftai_cold | claude-haiku-4-5 | 120 | 0 | 0 | 0 |
| support_triage | thriftai_warm | claude-haiku-4-5 | 0 | 120 | 0 | 0 |

## Per-workload deep dives

### support_triage

**Cost reduction per condition** (mean across seeds and any models; warm vs. baseline tells the headline savings):

| Condition | Paid mean | Saved mean | Reduction vs. baseline |
|---|---|---|---|
| baseline | $0.0003 | $0.0000 | +0.0% |
| thriftai_cold | $0.0003 | $0.0000 | +0.3% |
| thriftai_warm | $0.0000 | $0.0003 | +100.0% |

**Latency per condition** (p50 / p95 ms, all calls included):

| Condition | p50 | p95 |
|---|---|---|
| baseline | 772 | 2202 |
| thriftai_cold | 762 | 1972 |
| thriftai_warm | 0 | 1 |

**Quality (Opus judge, 1-5 mean ± std):**

| Condition | Score |
|---|---|
| baseline | 4.47 ± 0.34 |
| thriftai_cold | 4.47 ± 0.39 |
| thriftai_warm | 4.42 ± 0.35 |


## Methodology

See `benchmarks/README.md` and `benchmarks/PLAN.md`.

## Raw data

Per-call records under `benchmarks/results/raw/<run_id>/calls.jsonl`.
Every dollar figure above is derived from raw token counts in those
files multiplied by `benchmarks/pricing.yaml`; see `make rederive`
for the verification script.
