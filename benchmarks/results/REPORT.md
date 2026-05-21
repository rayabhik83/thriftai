# ThriftAI Benchmark Results

> **Status: complete.** All planned workloads measured.
>
> Generated 2026-05-21 02:37 UTC from 1710 calls across 30 run(s).
> Pricing snapshot: pulled 2026-05-19 — [source](https://www.anthropic.com/pricing#anthropic-api).

## Headline

| Workload | Condition | Model | $/task paid (mean ± std) | $/task saved (mean ± std) | Quality (1-5, mean ± std) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|---|
| code_review | baseline | claude-haiku-4-5 | 0.0017 ± 0.0004 $ | 0.0000 ± 0.0000 $ | — | 2812 | 4409 |
| code_review | thriftai_cold | claude-haiku-4-5 | 0.0017 ± 0.0003 $ | 0.0000 ± 0.0000 $ | — | 2823 | 4196 |
| code_review | thriftai_warm | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0017 ± 0.0004 $ | — | 0 | 1 |
| humaneval | baseline | claude-haiku-4-5 | 0.0002 ± 0.0001 $ | 0.0000 ± 0.0000 $ | 5.00 ± 0.00 | 1290 | 2208 |
| humaneval | thriftai_cold | claude-haiku-4-5 | 0.0002 ± 0.0001 $ | 0.0000 ± 0.0000 $ | 4.90 ± 0.63 | 1193 | 2070 |
| humaneval | thriftai_warm | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0002 ± 0.0001 $ | 5.00 ± 0.00 | 0 | 1 |
| research_analyst | baseline | claude-haiku-4-5 | 0.0018 ± 0.0002 $ | 0.0000 ± 0.0000 $ | 4.44 ± 0.18 | 3584 | 8460 |
| research_analyst | thriftai_cold | claude-haiku-4-5 | 0.0018 ± 0.0002 $ | 0.0000 ± 0.0000 $ | 4.46 ± 0.19 | 3703 | 7629 |
| research_analyst | thriftai_replay | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0018 ± 0.0002 $ | 4.47 ± 0.16 | 0 | 0 |
| research_analyst | thriftai_warm | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0019 ± 0.0003 $ | 4.44 ± 0.17 | 0 | 1 |
| support_triage | baseline | claude-haiku-4-5 | 0.0003 ± 0.0000 $ | 0.0000 ± 0.0000 $ | 4.47 ± 0.34 | 772 | 2202 |
| support_triage | baseline | claude-sonnet-4-6 | 0.0038 ± 0.0005 $ | 0.0000 ± 0.0000 $ | — | 1275 | 3896 |
| support_triage | thriftai_cold | claude-haiku-4-5 | 0.0003 ± 0.0000 $ | 0.0000 ± 0.0000 $ | 4.47 ± 0.39 | 762 | 1972 |
| support_triage | thriftai_cold | claude-sonnet-4-6 | 0.0039 ± 0.0003 $ | 0.0000 ± 0.0000 $ | — | 1585 | 4023 |
| support_triage | thriftai_warm | claude-haiku-4-5 | 0.0000 ± 0.0000 $ | 0.0003 ± 0.0000 $ | 4.42 ± 0.35 | 0 | 1 |
| support_triage | thriftai_warm | claude-sonnet-4-6 | 0.0000 ± 0.0000 $ | 0.0039 ± 0.0003 $ | — | 0 | 0 |

## Call resolution breakdown

Counts of brokered-call outcomes per cell. Cache vs replay vs live
tells you which mechanism is doing the work.

| Workload | Condition | Model | live | cache_hit | semantic_hit | replay |
|---|---|---|---:|---:|---:|---:|
| code_review | baseline | claude-haiku-4-5 | 120 | 0 | 0 | 0 |
| code_review | thriftai_cold | claude-haiku-4-5 | 120 | 0 | 0 | 0 |
| code_review | thriftai_warm | claude-haiku-4-5 | 0 | 120 | 0 | 0 |
| humaneval | baseline | claude-haiku-4-5 | 40 | 0 | 0 | 0 |
| humaneval | thriftai_cold | claude-haiku-4-5 | 40 | 0 | 0 | 0 |
| humaneval | thriftai_warm | claude-haiku-4-5 | 0 | 40 | 0 | 0 |
| research_analyst | baseline | claude-haiku-4-5 | 160 | 0 | 0 | 0 |
| research_analyst | thriftai_cold | claude-haiku-4-5 | 160 | 0 | 0 | 0 |
| research_analyst | thriftai_replay | claude-haiku-4-5 | 0 | 40 | 0 | 120 |
| research_analyst | thriftai_warm | claude-haiku-4-5 | 0 | 160 | 0 | 0 |
| support_triage | baseline | claude-haiku-4-5 | 120 | 0 | 0 | 0 |
| support_triage | baseline | claude-sonnet-4-6 | 110 | 0 | 0 | 0 |
| support_triage | thriftai_cold | claude-haiku-4-5 | 120 | 0 | 0 | 0 |
| support_triage | thriftai_cold | claude-sonnet-4-6 | 60 | 0 | 0 | 0 |
| support_triage | thriftai_warm | claude-haiku-4-5 | 0 | 120 | 0 | 0 |
| support_triage | thriftai_warm | claude-sonnet-4-6 | 0 | 60 | 0 | 0 |

## Per-workload deep dives

### code_review

**Cost reduction per condition** (mean across seeds and any models; warm vs. baseline tells the headline savings):

| Condition | Paid mean | Saved mean | Reduction vs. baseline |
|---|---|---|---|
| baseline | $0.0017 | $0.0000 | +0.0% |
| thriftai_cold | $0.0017 | $0.0000 | -0.1% |
| thriftai_warm | $0.0000 | $0.0017 | +100.0% |

**Latency per condition** (p50 / p95 ms, all calls included):

| Condition | p50 | p95 |
|---|---|---|
| baseline | 2812 | 4409 |
| thriftai_cold | 2823 | 4196 |
| thriftai_warm | 0 | 1 |

**Quality (Opus judge, 1-5 mean ± std):**

| Condition | Score |
|---|---|
| baseline | — |
| thriftai_cold | — |
| thriftai_warm | — |

### humaneval

**Cost reduction per condition** (mean across seeds and any models; warm vs. baseline tells the headline savings):

| Condition | Paid mean | Saved mean | Reduction vs. baseline |
|---|---|---|---|
| baseline | $0.0002 | $0.0000 | +0.0% |
| thriftai_cold | $0.0002 | $0.0000 | +0.0% |
| thriftai_warm | $0.0000 | $0.0002 | +100.0% |

**Latency per condition** (p50 / p95 ms, all calls included):

| Condition | p50 | p95 |
|---|---|---|
| baseline | 1290 | 2208 |
| thriftai_cold | 1193 | 2070 |
| thriftai_warm | 0 | 1 |

**Quality (Opus judge, 1-5 mean ± std):**

| Condition | Score |
|---|---|
| baseline | 5.00 ± 0.00 |
| thriftai_cold | 4.90 ± 0.63 |
| thriftai_warm | 5.00 ± 0.00 |

### research_analyst

**Cost reduction per condition** (mean across seeds and any models; warm vs. baseline tells the headline savings):

| Condition | Paid mean | Saved mean | Reduction vs. baseline |
|---|---|---|---|
| baseline | $0.0018 | $0.0000 | +0.0% |
| thriftai_cold | $0.0018 | $0.0000 | +1.3% |
| thriftai_replay | $0.0000 | $0.0018 | +100.0% |
| thriftai_warm | $0.0000 | $0.0019 | +100.0% |

**Latency per condition** (p50 / p95 ms, all calls included):

| Condition | p50 | p95 |
|---|---|---|
| baseline | 3584 | 8460 |
| thriftai_cold | 3703 | 7629 |
| thriftai_replay | 0 | 0 |
| thriftai_warm | 0 | 1 |

**Quality (Opus judge, 1-5 mean ± std):**

| Condition | Score |
|---|---|
| baseline | 4.44 ± 0.18 |
| thriftai_cold | 4.46 ± 0.19 |
| thriftai_replay | 4.47 ± 0.16 |
| thriftai_warm | 4.44 ± 0.17 |

### support_triage

**Cost reduction per condition** (mean across seeds and any models; warm vs. baseline tells the headline savings):

| Condition | Paid mean | Saved mean | Reduction vs. baseline |
|---|---|---|---|
| baseline | $0.0020 | $0.0000 | +0.0% |
| thriftai_cold | $0.0015 | $0.0000 | +24.7% |
| thriftai_warm | $0.0000 | $0.0015 | +100.0% |

**Latency per condition** (p50 / p95 ms, all calls included):

| Condition | p50 | p95 |
|---|---|---|
| baseline | 1137 | 3666 |
| thriftai_cold | 1137 | 3602 |
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
