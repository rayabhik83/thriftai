# ThriftAI Benchmark Suite

Reproducible benchmarks for ThriftAI's caching and replay features. Designed to satisfy a skeptical-senior-engineer level of scrutiny.

> Status: **scaffolding in progress**. See `benchmarks/PLAN.md` for the full build plan and current step.

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
