# ThriftAI Benchmark Suite
#
# Targets:
#   make install   — pip install -e ".[bench]" (from repo root)
#   make smoke     — N=2, support_triage only, Haiku only (~$0.10 of API)
#   make bench     — N=5, all workloads, Haiku + Sonnet (real money)
#   make report    — regenerate REPORT.md from existing raw logs (free)
#   make rederive  — recompute $ figures from raw logs (sanity check)
#   make clean     — remove raw logs and bench cache (does NOT delete REPORT.md)
#
# Requires Python 3.11+ for the benchmark suite.
# Requires ANTHROPIC_API_KEY in benchmarks/.env (see .env.example).

PY ?= python3
REPO_ROOT := $(shell git rev-parse --show-toplevel)

.PHONY: help install test smoke bench report rederive judge plots clean spent budget-reset

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sed -e 's/:.*##/ — /'

install: ## install bench extras
	cd $(REPO_ROOT) && $(PY) -m pip install -e ".[bench]"

test: ## run the bench harness tests (no API calls)
	cd $(REPO_ROOT) && $(PY) -m pytest benchmarks/tests/ -q

smoke: ## N=2, support_triage (20 tasks), Haiku only — fast validation
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.run \
		--workload support_triage \
		--n 2 \
		--task-limit 20 \
		--model claude-haiku-4-5

bench: ## N=5, all workloads, Haiku + Sonnet
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.run \
		--workload all \
		--n 5

report: ## regenerate REPORT.md from existing raw logs (free)
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.report

rederive: ## verify dollar figures from raw logs + pricing.yaml
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.rederive

judge: ## score every artifacts.jsonl with Opus and write judge_scores.jsonl
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.judge_runner

humaneval-score: ## run official human_eval evaluator on humaneval artifacts
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.humaneval_score

plots: ## render the 3 required figures into benchmarks/results/plots/
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.plots

clean: ## clear raw logs and bench cache (keeps REPORT.md and plots/)
	rm -rf $(REPO_ROOT)/benchmarks/results/raw $(REPO_ROOT)/benchmarks/cache

spent: ## print total recorded spend (USD) across all runs on this branch
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.budget total

budget-reset: ## clear the persistent spend ledger
	cd $(REPO_ROOT) && $(PY) -m benchmarks.runner.budget reset
