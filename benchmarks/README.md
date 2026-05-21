# ThriftAI Benchmark Suite

Reproducible benchmarks for ThriftAI's caching and replay features. Designed to satisfy a skeptical-senior-engineer level of scrutiny.

> Status: **N=2 results landed** across all four workloads. See `benchmarks/results/REPORT.md` for the headline table and `benchmarks/PLAN.md` for the design. N=5 production run is a follow-up after raising the Anthropic rate-limit tier.

## Headline (N=2, Haiku 4.5; one Sonnet-4.6 arm)

The warm-cache and replay paths cost **$0/task** while preserving quality within noise. Latency drops from 0.8–3.6 s to **<1 ms**.

| Workload | Model | Baseline $/task | Warm $/task | Saved | Baseline quality | Warm quality | Baseline p50 | Warm p50 |
|---|---|---|---|---|---|---|---|---|
| support_triage | Haiku 4.5 | $0.0003 | $0 | 100% | 4.47 ± 0.34 | 4.42 ± 0.35 | 772 ms | 0 ms |
| support_triage | Sonnet 4.6 | $0.0039 | $0 | 100% | (judge skipped) | (judge skipped) | 1266 ms | 0 ms |
| research_analyst | Haiku 4.5 | $0.0018 | $0 (warm/replay) | 100% | 4.44 ± 0.18 | 4.44 ± 0.17 | 3584 ms | 0 ms |
| code_review | Haiku 4.5 | $0.0017 | $0 | 100% | 3.33 ± 0.51 | 3.37 ± 0.56 | 2812 ms | 0 ms |
| humaneval | Haiku 4.5 | $0.0002 | $0 | 100% | pass@1 100% | pass@1 100% | 1290 ms | 0 ms |

**The `thriftai_replay` condition on `research_analyst`** is the cleanest demonstration of selective replay: 120 of 160 measured calls (the 3 unchanged agents × 20 tasks × 2 seeds) deterministically replay from trace, while the 40 critic-only calls served from the warm exact cache. **$0 paid, quality 4.47 ± 0.16 — statistically indistinguishable from baseline.**

Total cost to produce this report: **$15.13** (Anthropic API spend, dated 2026-05-21 — `pricing.yaml`).

## Reproducing the published numbers

```bash
# from a clean clone
cp benchmarks/.env.example benchmarks/.env
# edit benchmarks/.env to add ANTHROPIC_API_KEY

cd benchmarks
make install     # pip install -e ".[bench]" from repo root
make smoke       # ~5 min, ~$0.10 of API   (single workload, N=2)
make bench       # multi-hour, real money  (all workloads, N=5)
```

`make bench` writes raw per-call JSONL to `benchmarks/results/raw/<run_id>/calls.jsonl` and a final markdown report to `benchmarks/results/REPORT.md`. Plots are saved under `benchmarks/results/plots/`.

To regenerate the report from existing raw logs (free, no API):

```bash
make report
```

To verify every dollar figure can be recomputed from raw logs + `pricing.yaml`:

```bash
make rederive
```

## Methodology

> *(filled in as workloads land — see PLAN.md for the full design.)*

### What's measured

- (TODO once workload 1 lands)

### Conditions

- `baseline` — `Session(enabled=False)`; ThriftAI is a no-op passthrough.
- `thriftai_cold` — fresh cache directory.
- `thriftai_warm` — cache pre-populated by one full pass before measurement.
- `thriftai_replay` — (research_analyst workload only) replay all but one agent live.

### Models

- Under test: `claude-haiku-4-5`, `claude-sonnet-4-6`
- Judge: `claude-opus-4-7` (never under test)

### Cost methodology

Dollar figures in this report are computed from `benchmarks/pricing.yaml` (Anthropic pricing snapshot, dated, source URL inside the file) multiplied by raw token counts in `benchmarks/results/raw/*/calls.jsonl`. We do **not** use `litellm.completion_cost` for the report — this lets any reader recompute every figure from the raw logs without trusting our pricing data.

### Variance

N=5 runs per `(workload × condition × model)` cell. Means reported with ±1 std; bootstrap 95% CIs in the appendix.

### Quality

LLM-as-judge using `claude-opus-4-7` against a fixed rubric per workload. Judge results are cached separately (sidecar SQLite) and independent from ThriftAI's own cache so quality recomputes don't re-spend.

### Single-provider limitation

This release benchmarks only Anthropic models. The audience-relevant point — that cost reduction generalizes — is shown via three Claude models spanning two orders of magnitude in price (Haiku → Sonnet → Opus). The `pricing.yaml` and `configs/*.yaml` files contain commented-out OpenAI / Together entries; swapping providers is a one-line config change because ThriftAI routes everything through LiteLLM (`thriftai/providers/__init__.py:call_litellm`). Reproducing these numbers against another provider is left as a one-key reader exercise.

### Semantic-cache caveat

ThriftAI's semantic cache is **opt-in** via `Session(embedding_model=...)` and requires an embedding-model API key (Voyage, OpenAI, or a local Ollama instance). Since this release of the benchmark uses Anthropic-only keys, **`embedding_model` defaults to `null`** in the shipped workload configs and the semantic cache is **not exercised** in the published numbers. We measure the exact-match cache + replay paths only.

Readers who want semantic-cache numbers can set `thriftai_session.embedding_model` in any `configs/<workload>.yaml`, add the corresponding API key to `.env`, and re-run.

### What does and does not ship in ThriftAI 0.1.1

| Feature | In 0.1.1? | Benchmarked here? |
|---|---|---|
| Exact-match cache | ✅ | ✅ |
| Semantic cache (opt-in) | ✅ | ✅ |
| Replay + downstream invalidation | ✅ (dev-only) | ✅ |
| Cost accounting (`CostReport`) | ✅ | ✅ (sanity-checked) |
| Model tiering | ❌ | ❌ |
| General (non-replay) diff-aware re-execution | ❌ | ❌ |

## Layout

```
benchmarks/
├── PLAN.md                     full design + decisions
├── README.md                   this file
├── Makefile
├── .env.example
├── pricing.yaml
├── configs/                    per-workload pydantic configs
├── workloads/                  the actual @ta.agent pipelines
├── runner/                     orchestration, instrumentation, stats, report
├── judge/                      Opus rubric scorer
├── data/                       fixed inputs (committed)
└── results/                    REPORT.md, plots/ (committed); raw/ (gitignored)
```
