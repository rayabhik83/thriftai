# ThriftAI Benchmark Suite — Build Plan

> Status: **approved (2026-05-19)**. Scaffolding work in progress; see the "Build order" section for current step.

## What this is

A reproducible benchmark suite that demonstrates ThriftAI's cost reduction and quality impact on realistic multi-agent workloads. The bar is research-paper rigor: N≥3 runs with variance, explicit baselines, quality numbers alongside every cost number, dollar-recomputable raw logs, and a one-command path from clean clone to final report.

The audience is a skeptical senior engineer reading the README on Hacker News. If they want to challenge a number, the raw logs should let them recompute it themselves.

## What gets benchmarked

Only features that **ship in ThriftAI 0.1.1**:

| Feature | Source file | Measured as |
|---|---|---|
| Exact-match cache | `thriftai/cache/__init__.py` | hit rate, cost reduction, lookup latency |
| Semantic cache (opt-in) | `thriftai/cache/semantic.py` | hit rate, embedding overhead, wrong-hit rate |
| Replay + downstream invalidation | `thriftai/session.py:ReplayContext` | selective-replay cost vs full live |
| Cost accounting | `thriftai/cost/__init__.py` | sanity-checked against our own per-call instrumentation |

**Out of scope for this release:** model tiering and general (non-replay) diff-aware re-execution. Neither ships yet. The methodology section calls them out explicitly so readers don't expect numbers we can't honestly produce.

## Provider strategy

Single provider: **Anthropic**. The brief's "three model families" criterion is satisfied with three Claude models spanning the cost/capability spectrum:

| Role | Model |
|---|---|
| Frontier under test | `claude-sonnet-4-6` |
| Cheap under test | `claude-haiku-4-5` |
| Judge only (never under test) | `claude-opus-4-7` |

The single-provider limitation is documented in `benchmarks/README.md`. The runner config has commented-out OpenAI/Together entries so any reader can swap providers with one config change and rerun against their own keys — `thriftai/providers/__init__.py:call_litellm` is the single provider abstraction, which makes this trivial in practice.

## Methodology

### Conditions (every workload runs all three)

1. **`baseline`** — `Session(enabled=False)`. ThriftAI is a no-op passthrough to LiteLLM. Same prompts, same models, same temperature, same seeds where supported.
2. **`thriftai_cold`** — `Session(enabled=True, embedding_model=<embed>)` with a fresh `cache_dir`. Models the day-one user experience.
3. **`thriftai_warm`** — same as cold, but `cache_dir` is pre-populated by one full pass over the workload before the measured run. Models steady-state savings.

Workload 2 (research analyst) adds a fourth condition:

4. **`thriftai_replay`** — run the full pipeline live once, then replay all but the `critic` agent live. Models the iterate-on-the-last-agent dev workflow.

### Variance

- **N = 5** runs per `(workload × condition × model)` cell.
- Mean ± 1 std reported in headline tables.
- Bootstrap 95% CIs computed by `runner/stats.py` and shown in the appendix.
- If a cell is judged too expensive during development, N may drop to 3 with the rationale documented in the report (not silently).

### Per-call instrumentation

`runner/instrumentation.py` monkey-patches `thriftai.providers.call_litellm` (no edit to the library). Every API call writes a line to `benchmarks/results/raw/<run_id>/calls.jsonl`:

```json
{
  "timestamp": "2026-05-19T22:00:00Z",
  "run_id": "20260519_220000_support_triage_cold_haiku_seed3",
  "workload": "support_triage",
  "condition": "thriftai_cold",
  "model": "claude-haiku-4-5",
  "agent_name": "classifier",
  "task_id": "ticket_017",
  "input_tokens": 412,
  "output_tokens": 38,
  "latency_ms": 873,
  "broker_resolution": "live",
  "response_text_hash": "sha256:abc...",
  "seed": 3
}
```

`broker_resolution` is sourced from `thriftai.broker.CallResolution` (`live`, `cache_hit`, `semantic_hit`, `replay`).

### Cost computation — independent from LiteLLM

Dollar figures are computed in our own code from `benchmarks/pricing.yaml` × raw token counts, **not** from `litellm.completion_cost`. `pricing.yaml` has dated entries (pulled from Anthropic's pricing page on a specific date documented in the file) and is the single source of truth for the report's dollar numbers.

This means: any reader can re-derive every dollar figure from `calls.jsonl + pricing.yaml`. A `runner/rederive.py` script does exactly this and is committed alongside.

### Latency

- End-to-end wall-clock per task: p50 and p95 over N runs.
- Per-call latency from the JSONL.
- Report breaks total latency into:
  - **ThriftAI overhead** — broker decision time, cache lookup, embedding round-trip (semantic only)
  - **Underlying API time** — LiteLLM call duration

Overhead on a miss is reported, not hidden.

### Quality

LLM-as-judge using `claude-opus-4-7` against a strict rubric per workload. Judge calls are cached separately by `hash(task_id, output_text)` in a sidecar SQLite (independent of ThriftAI's own cache) so quality recomputes don't re-spend.

Where a deterministic metric exists alongside the judge (e.g. exact-match on a structured field, pass@1 for HumanEval), both are reported. Deterministic > LLM judge whenever both are available.

### Cache hit rate broken out

The report distinguishes:

- exact-cache hits
- semantic-cache hits (with mean similarity score on hits)
- replay hits (workload 2 replay condition only)

A single "cache hit rate" number is misleading because each layer has different reliability properties.

## Workloads

All workloads use `@ta.agent`-decorated functions so they exercise the real public API. Pattern is the same as `examples/research_pipeline.py`.

### 1. Support triage (built first)

**Pipeline:** ticket → `classifier` → `retriever` (cosine over the ticket corpus) → `drafter`.

**Inputs:** 50 tickets synthesized by `claude-sonnet-4-6` from a written rubric covering ~10 "recurring complaint" clusters with paraphrastic variants. Repetition is the point — exact and semantic caches should both shine. Both the synthesis script (`data/support_tickets_gen.py`) **and** its output (`data/support_tickets.jsonl`) are committed; the JSONL is the canonical input for benchmark runs and the script is only re-run if we want to regenerate.

**Quality rubric (Opus, 1–5):** classification correctness, retrieval relevance, draft helpfulness. Averaged per ticket.

**Headline numbers:** cost-per-ticket × {baseline / cold / warm} × Haiku and Sonnet, with cache-hit breakdown.

### 2. Research analyst pipeline

**Pipeline:** question → `scout` → `planner` (`depends_on=["scout"]`) → `analyst` (`depends_on=["planner"]`) → `critic` (`depends_on=["analyst"]`).

**Inputs:** 20 fixed research questions, intentionally diverse across finance/science/history/current-events so semantic-cache wrong-hit risk gets exercised.

**Quality rubric (Opus, 1–5):** factual accuracy (where checkable), plan coherence, analysis depth, critique sharpness.

**Headline numbers:** baseline / cold / warm / **replay** (`live=["critic"]`). Per-agent hit-rate breakdown.

### 3. Code review loop

**Pipeline:** diff → `reviewer` → `proposer` → `self_critic`.

**Inputs:** 20 fixed code diffs sampled from `bigcode/the-stack-smol-xs` (Hugging Face), filtered to functions of a tractable size. Sampling script (`data/code_review_sample.py`) and the resulting JSONL are both committed. The JSONL is canonical; the sampling script is for reproducibility / regeneration only.

**Quality:** LLM judge for review quality + (where ground-truth fixes exist) exact-match on the fix region.

**Headline numbers:** baseline / cold / warm, latency overhead emphasized.

### 4. HumanEval slice (public benchmark)

20-problem slice scored via the official `human-eval` package (`pass@1`). Demonstrates ThriftAI doesn't perturb correctness on a benchmark the audience already trusts.

SWE-bench Lite was the brief's preferred option; deferred to a follow-up PR because its Docker-per-task harness blows up the smoke-test path. HumanEval is the brief's accepted fallback.

## File layout

```
benchmarks/
├── PLAN.md                          (this file)
├── README.md                        methodology + reproduction
├── Makefile                         make {bench,report,smoke,install}
├── .env.example                     ANTHROPIC_API_KEY=
├── pricing.yaml                     dated pricing per model
├── configs/
│   ├── research_analyst.yaml
│   ├── code_review.yaml
│   ├── support_triage.yaml
│   └── humaneval.yaml
├── workloads/
│   ├── __init__.py
│   ├── support_triage.py
│   ├── research_analyst.py
│   ├── code_review.py
│   └── humaneval.py
├── runner/
│   ├── __init__.py
│   ├── run.py                       entry point: orchestrate conditions × N
│   ├── conditions.py                baseline / cold / warm / replay
│   ├── instrumentation.py           monkey-patch + JSONL writer
│   ├── stats.py                     mean/std/bootstrap CI
│   ├── report.py                    JSONL → REPORT.md + plots
│   └── rederive.py                  recompute $ from raw JSONL + pricing.yaml
├── judge/
│   └── llm_judge.py                 Opus-based rubric scorer
├── data/
│   ├── research_questions.jsonl
│   ├── support_tickets.jsonl
│   ├── support_tickets_gen.py       synthesis script (output committed)
│   └── code_review_prs.jsonl
├── results/                         REPORT.md + plots committed; raw/ gitignored
│   ├── REPORT.md
│   ├── plots/
│   └── raw/                         (gitignored)
└── cache/                           (gitignored — bench artifact cache + judge cache)
```

## Dependency management

Add `[bench]` to root `pyproject.toml` `[project.optional-dependencies]`, matching the existing `[dev]`, `[semantic]`, `[docs]` pattern.

New deps:

- `matplotlib` — plots
- `pyyaml` — config + pricing files
- `python-dotenv` — `.env` loading
- `scipy` — bootstrap CIs
- `human-eval` — public benchmark

Already-present deps reused: `litellm`, `pydantic`, `pytest`, `numpy` (via `[semantic]`).

Python 3.11+ for the benchmark suite (3.10+ for the main lib stays unchanged). The version floor is enforced in the Makefile's `install` target.

## Reproduction

```bash
# from clean clone
cp benchmarks/.env.example benchmarks/.env
# edit benchmarks/.env to add ANTHROPIC_API_KEY
make smoke        # ~5 min, ~$0.10 of API
# or, for the published numbers:
make bench        # ~hours, real money
```

Makefile targets:

```makefile
install:                                 # pip install -e ".[bench]"
smoke:    # N=2, support_triage only, Haiku only       (~$0.10)
bench:    # N=5, all workloads, Haiku + Sonnet         (real money)
report:   # regenerate REPORT.md from existing raw logs (free)
```

## Build order (strict)

Each step is a separate commit. Each gates on the previous step being green and on me being satisfied that the work matches this plan.

1. **PLAN.md committed and reviewed** ← we are here
2. **Scaffolding** — directory tree, Makefile skeleton, `[bench]` extras, `.env.example`, `pricing.yaml`, empty README with section headers. No workload code.
3. **Instrumentation harness** — `runner/instrumentation.py` end-to-end verified against a hardcoded prompt.
4. **Stats + empty report skeleton** — `runner/stats.py` and `runner/report.py` render an empty headline table without crashing.
5. **Workload 1 (support triage) end-to-end** — N=2, Haiku only, all 3 conditions. First time the report has real numbers.
6. **Judge wired** — Opus 1–5 rubric + judge cache. Re-run workload 1 with quality numbers.
7. **Workload 2 (research analyst)** — includes the replay condition.
8. **Workload 3 (code review).**
9. **HumanEval slice.**
10. **Multi-model axis** — add Sonnet. Bump N to 5.
11. **Plots** — three required figures.
12. **Clean-clone smoke test** — `git clone /tmp/x && cd /tmp/x && make smoke` works without manual steps beyond the API key.
13. **Final REPORT.md** committed. Open PR.

## Decisions resolved during review

1. **Code-review diffs** — sampled from `bigcode/the-stack-smol-xs`; sampling script and resulting JSONL both committed.
2. **Support-triage corpus** — synthesized by `claude-sonnet-4-6`; synthesis script and resulting JSONL both committed.
3. **`benchmarks/results/` git policy** — commit `REPORT.md` and `plots/*.png`; gitignore `raw/`.
4. **API keys** — single `ANTHROPIC_API_KEY` for under-test and judge calls. Judge spend is itemized in the report by model, which is sufficient for accounting.
5. **CI integration** — `.github/workflows/bench-smoke.yml` runs `make smoke` on every PR. Requires `ANTHROPIC_API_KEY` to be set as a GitHub repo secret. Workflow skips with a clear message when the secret isn't available (so forked-PR CI doesn't try to use a key it can't see).

## Verification (criteria for the final PR, not for this plan)

- `git clone && cp .env.example .env && # add ANTHROPIC_API_KEY && make smoke` works on a clean machine in <5 minutes.
- `make bench` reproduces the headline numbers in the README within ±1 std of the published values.
- `benchmarks/results/REPORT.md` renders cleanly on GitHub, plots embedded.
- Every dollar figure in the report is recomputed from `benchmarks/results/raw/*.jsonl × pricing.yaml` by `runner/rederive.py` and matches.
- `ruff check benchmarks/` is clean.
- A pytest smoke run inside `benchmarks/` passes.
- The methodology section in `benchmarks/README.md` explicitly states: single-provider limitation, judge model, pricing date, N runs, what does and doesn't ship in ThriftAI 0.1.1.
