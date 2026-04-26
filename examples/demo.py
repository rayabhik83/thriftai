"""
ThriftAI CLI demo — same 3-agent pipeline as `examples/demo.ipynb`,
terminal output. Designed for screen-recording (asciinema, etc.) so the
value loop can be shared as a GIF.

Run:
    python examples/demo.py
    python examples/demo.py --topic "Why Rust matters" --model anthropic/claude-haiku-4-5

The default model is `anthropic/claude-haiku-4-5`; full demo cost is
under $0.05 with that model. Set ANTHROPIC_API_KEY before running.

Output is a 3-row cost table:

    1 (cold)        all 3 live        — every agent paid
    2 (re-run)      all 3 cache hit   — exact-match cache catches everything
    3 (writer fix)  replay 2, live 1  — selective replay, only writer paid

The "Live cost" column is what the run *would have* cost with no
ThriftAI; the "ThriftAI" column is what it *actually* cost.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import thriftai as ta

DEFAULT_TOPIC = "The impact of AI on cybersecurity"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
DEMO_DIR = Path("./.thriftai-demo")


def _build_pipeline(model: str):
    """Define the 3-agent pipeline against a chosen model.

    Defining inside a function (rather than at module scope) lets the
    --model CLI flag override the default without monkey-patching globals.
    """

    @ta.agent(name="researcher")
    def research(s, topic: str) -> str:
        return s.completion(
            messages=[
                {"role": "system", "content":
                    "You are a research assistant. Be concise. "
                    "Return a short paragraph of factual context."},
                {"role": "user", "content": f"Research this topic: {topic}"},
            ],
            model=model,
        )

    @ta.agent(name="analyzer", depends_on=["researcher"])
    def analyze(s, raw: str) -> str:
        return s.completion(
            messages=[
                {"role": "system", "content":
                    "You are an analyst. List the 3 most important "
                    "insights from the research, one per line."},
                {"role": "user", "content": f"Research:\n\n{raw}"},
            ],
            model=model,
        )

    @ta.agent(name="writer", depends_on=["analyzer"])
    def write(s, analysis: str, *, style: str) -> str:
        return s.completion(
            messages=[
                {"role": "system", "content":
                    f"You are a technical writer. Produce a {style} summary "
                    "in 3 short paragraphs."},
                {"role": "user", "content": f"Analysis:\n\n{analysis}"},
            ],
            model=model,
        )

    return research, analyze, write


def _print_table(rows: list[tuple[str, str, float, float]]) -> None:
    # Column widths: must stay in sync between fmt, header, sub-rule, and pct gutter.
    fmt = "  {:<15} {:<17} {:>10}  {:>10}  {:>9}"
    sub = (
        "  " + "─" * 15 + " " + "─" * 17 + " " + "─" * 10
        + "  " + "─" * 10 + "  " + "─" * 9
    )
    rule = "═" * len(sub)

    print(rule)
    print("  ThriftAI demo — 3-agent research pipeline, 3 iterations")
    print(rule)
    print(fmt.format("Iteration", "Action", "Live cost", "ThriftAI", "Saved"))
    print(sub)

    total_would = total_actual = 0.0
    for label, action, would, actual in rows:
        saved = would - actual
        total_would += would
        total_actual += actual
        print(fmt.format(
            label, action,
            f"${would:.4f}", f"${actual:.4f}", f"${saved:.4f}",
        ))

    print(sub)
    saved_total = total_would - total_actual
    print(fmt.format(
        "Total", "",
        f"${total_would:.4f}", f"${total_actual:.4f}", f"${saved_total:.4f}",
    ))
    if total_would > 0:
        pct = saved_total / total_would * 100
        # Right-align (NN%) under the Saved column.
        print(" " * (len(sub) - 4) + f"({pct:.0f}%)")
    print(rule)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="examples/demo.py",
        description="ThriftAI CLI demo — see cost savings across 3 iterations.",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help=f"Topic for the research pipeline (default: {DEFAULT_TOPIC!r})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Don't wipe ./.thriftai-demo/ before running.")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY") and "anthropic" in args.model:
        print(
            "ERROR: set ANTHROPIC_API_KEY (or pass --model with a different "
            "provider, e.g. --model openai/gpt-4o-mini and set OPENAI_API_KEY).",
            file=sys.stderr,
        )
        return 1

    if not args.keep_cache and DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)

    research, analyze, write = _build_pipeline(args.model)
    session = ta.Session(cache_dir=str(DEMO_DIR))
    rows: list[tuple[str, str, float, float]] = []

    # Run 1 — cold. All 3 agents go live.
    with session.run() as run:
        analysis = analyze(run, research(run, args.topic))
        summary = write(run, analysis, style="executive")
        trace_id_1 = run.trace_id
    rows.append((
        "1 (cold)", "all 3 live",
        sum(e.would_have_cost_usd for e in run.cost_report.entries),
        run.cost_report.total_cost,
    ))

    # Run 2 — identical inputs. Exact-match cache catches all 3.
    with session.run() as run:
        analysis = analyze(run, research(run, args.topic))
        summary = write(run, analysis, style="executive")
    rows.append((
        "2 (re-run)", "all 3 cache hit",
        sum(e.would_have_cost_usd for e in run.cost_report.entries),
        run.cost_report.total_cost,
    ))

    # Run 3 — change the writer's style; selective replay so only writer
    # makes a live call (researcher + analyzer come from the trace).
    with session.replay(trace_id=trace_id_1, live=["writer"]) as run:
        analysis = analyze(run, research(run, args.topic))
        summary = write(run, analysis, style="conversational")
    rows.append((
        "3 (writer fix)", "replay 2, live 1",
        sum(e.would_have_cost_usd for e in run.cost_report.entries),
        run.cost_report.total_cost,
    ))

    _print_table(rows)
    print("\nFinal summary:\n")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
