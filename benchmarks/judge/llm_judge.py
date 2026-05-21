"""
LLM-as-judge — scores workload artifacts on a 1-5 rubric using Opus.

Each workload defines its own rubric (system prompt + scoring fields).
The judge is intentionally independent from ThriftAI's broker so its
own caching and cost are tracked separately and can't be confused
with the system under test:

- Judge cache: SQLite at `benchmarks/cache/judge.db`, keyed by
  `hash(workload, task_id, artifacts)`. A re-run with the same
  artifacts hits the cache for free, no API call.
- Judge spend: recorded in the same persistent ledger as workload
  spend (`benchmarks/runner/budget.py`), so the $10 cap is global.

Judge calls are deterministic-ish (temperature=0). The judge model
is excluded from the under-test set — we never measure Opus's own
quality or cost as a workload, only use it as a referee.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
CACHE_DB = BENCH_DIR / "cache" / "judge.db"
JUDGE_MODEL = "claude-opus-4-7"

_db_lock = threading.Lock()

# Minimum delay AFTER each live judge call. Opus has its own per-minute
# rate limit (typically 50 RPM on lower tiers). Mirrors the runner's
# in-process throttle so we don't crash mid-run.
JUDGE_THROTTLE_GAP_SEC = 1.3


# ---- per-workload rubrics --------------------------------------------------

SUPPORT_TRIAGE_RUBRIC = """\
You are a strict, fair judge of a customer-support triage pipeline.

You are given:
- ticket_text: the customer's original ticket.
- classifier_output: the pipeline's predicted category.
- retriever_output: the pipeline's chosen similar past ticket IDs.
- drafter_output: the pipeline's draft response to the customer.

Score each on a 1-5 integer scale:

  classification_correct (1-5):
    5 = perfectly matches a sensible reading of the ticket
    3 = plausible but not the best fit
    1 = clearly wrong category

  retrieval_relevance (1-5):
    5 = retrieved tickets directly address the customer's issue
    3 = retrieved tickets are loosely related
    1 = retrieved tickets are unrelated

  draft_helpful (1-5):
    5 = a draft that could be sent as-is, addresses the issue concretely
    3 = a draft that needs editing but is on the right track
    1 = a draft that misunderstands the issue or would harm the customer

Output ONLY valid JSON, no other text, with this exact shape:
{"classification_correct": <int>, "retrieval_relevance": <int>, "draft_helpful": <int>, "rationale": "<1-2 sentence justification>"}
"""


RESEARCH_ANALYST_RUBRIC = """\
You are a strict, fair judge of a research-analyst pipeline.

You are given:
- question: the research question.
- scout: a list of sub-questions the scout proposed.
- plan: the research plan.
- analysis: the analyst's answer.
- critique: the critic's pushback.

Score each on a 1-5 integer scale:

  scout_quality (1-5):
    5 = sub-questions cover the key angles for the question
    3 = a reasonable starting list, misses some major angles
    1 = clearly off-topic or trivially shallow

  plan_coherence (1-5):
    5 = the plan would, if executed, produce a complete answer
    3 = the plan is sensible but vague or skips steps
    1 = the plan is confused or contradicts the question

  analysis_depth (1-5):
    5 = a substantive multi-step analysis that addresses the question
    3 = an analysis that's correct but surface-level
    1 = generic, evasive, or factually wrong

  critique_sharpness (1-5):
    5 = identifies a real counter-argument or unaddressed factor
    3 = a generic critique that could apply anywhere
    1 = empty or no real engagement with the analysis

Output ONLY valid JSON, no other text, with this exact shape:
{"scout_quality": <int>, "plan_coherence": <int>, "analysis_depth": <int>, "critique_sharpness": <int>, "rationale": "<1-2 sentence justification>"}
"""


CODE_REVIEW_RUBRIC = """\
You are a strict, fair judge of a code-review pipeline.

You are given:
- code: the original snippet under review.
- issues: the reviewer's list of issues.
- patch: the proposer's suggested patch.
- critique: the self-critic's own follow-up critique.

Score each on a 1-5 integer scale:

  review_thoroughness (1-5):
    5 = surfaces real bugs / design issues / performance concerns
    3 = surface-level (style) but accurate
    1 = mostly invented issues or none

  patch_correctness (1-5):
    5 = patch addresses the top issues without introducing regressions
    3 = partially addresses but with minor problems
    1 = does not compile or breaks the original behavior

  self_critique_value (1-5):
    5 = identifies a real regression / missed edge case in the patch
    3 = a generic concern, partially valid
    1 = vacuous

Output ONLY valid JSON, no other text, with this exact shape:
{"review_thoroughness": <int>, "patch_correctness": <int>, "self_critique_value": <int>, "rationale": "<1-2 sentence justification>"}
"""


RUBRICS: dict[str, str] = {
    "support_triage": SUPPORT_TRIAGE_RUBRIC,
    "research_analyst": RESEARCH_ANALYST_RUBRIC,
    "code_review": CODE_REVIEW_RUBRIC,
}


# ---- judge cache (SQLite, sidecar) ----------------------------------------


def _init_db() -> sqlite3.Connection:
    """Open / create the judge cache. Each caller closes its own connection."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS judge_cache (
            key TEXT PRIMARY KEY,
            workload TEXT NOT NULL,
            task_id TEXT NOT NULL,
            score_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    return conn


def _cache_key(workload: str, task_id: str, artifacts: dict) -> str:
    serialized = json.dumps(
        {"workload": workload, "task_id": task_id, "artifacts": artifacts},
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_cached(workload: str, task_id: str, artifacts: dict) -> dict | None:
    key = _cache_key(workload, task_id, artifacts)
    with _db_lock:
        conn = _init_db()
        try:
            row = conn.execute(
                "SELECT score_json FROM judge_cache WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return json.loads(row[0])


def _put_cached(workload: str, task_id: str, artifacts: dict, scores: dict) -> None:
    key = _cache_key(workload, task_id, artifacts)
    with _db_lock:
        conn = _init_db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO judge_cache (key, workload, task_id, score_json) "
                "VALUES (?, ?, ?, ?)",
                (key, workload, task_id, json.dumps(scores)),
            )
            conn.commit()
        finally:
            conn.close()


# ---- core judge function --------------------------------------------------


def _parse_judge_response(text: str) -> dict:
    """Pull the first JSON object out of the judge's reply."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json fences if present.
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    return json.loads(text)


def judge(
    workload: str,
    task_id: str,
    artifacts: dict,
    *,
    model: str = JUDGE_MODEL,
    litellm_completion=None,
) -> dict:
    """Score one task's artifacts. Returns the parsed score dict.

    `litellm_completion` lets tests inject a fake; production code passes
    None and we lazy-import `litellm.completion`.
    """
    cached = _get_cached(workload, task_id, artifacts)
    if cached is not None:
        return cached

    rubric = RUBRICS.get(workload)
    if rubric is None:
        raise ValueError(f"No rubric defined for workload: {workload}")

    if litellm_completion is None:
        import litellm

        # Enable retry-with-backoff if not already configured (idempotent).
        litellm.num_retries = max(getattr(litellm, "num_retries", 0) or 0, 5)
        litellm_completion = litellm.completion

    # Opus 4.7 rejects the `temperature` parameter as deprecated. Other
    # Claude models still accept it. Conditional pass keeps the judge
    # deterministic-ish on older models without breaking newer ones.
    kwargs = {} if "opus-4-7" in model else {"temperature": 0.0}

    response = litellm_completion(
        model=model,
        messages=[
            {"role": "system", "content": rubric},
            {
                "role": "user",
                "content": json.dumps(
                    {"task_id": task_id, **artifacts},
                    sort_keys=True,
                ),
            },
        ],
        **kwargs,
    )
    raw_text = response.choices[0].message.content
    scores = _parse_judge_response(raw_text)

    # Throttle after each live judge call so successive Opus calls stay
    # under the per-minute rate limit.
    if JUDGE_THROTTLE_GAP_SEC > 0:
        time.sleep(JUDGE_THROTTLE_GAP_SEC)

    # Spend tracking for the judge — record alongside workload spend so
    # the same $10 cap covers both.
    try:
        from benchmarks.runner import budget as _budget

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        # Hardcoded fall-through pricing if we can't load pricing.yaml here.
        # The runner's instrumentation has the canonical pricing path; for
        # the judge we approximate with Opus public pricing.
        opus_input = 15.0 / 1_000_000.0
        opus_output = 75.0 / 1_000_000.0
        amount = input_tokens * opus_input + output_tokens * opus_output
        _budget.record(amount, run_id="judge", condition="judge")
    except Exception:
        # Spend tracking is best-effort; never let it abort a judge run.
        pass

    _put_cached(workload, task_id, artifacts, scores)
    return scores


def judge_artifacts_file(
    artifacts_path: Path,
    workload: str,
    *,
    model: str = JUDGE_MODEL,
) -> list[dict]:
    """Score every artifact in a JSONL file. Returns the per-task score list."""
    results: list[dict] = []
    with artifacts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_id = entry["task_id"]
            # The judge expects the artifact dict minus the task_id, so each
            # workload's expected artifact keys end up in the prompt.
            artifacts = {k: v for k, v in entry.items() if k != "task_id"}
            scores = judge(workload, task_id, artifacts, model=model)
            results.append({"task_id": task_id, **scores})
    return results
