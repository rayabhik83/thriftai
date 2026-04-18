"""
Example: 3-agent research pipeline with ThriftAI.

This is the README demo scenario:
- Agent 1 (researcher): gathers raw information
- Agent 2 (analyzer): analyzes the research
- Agent 3 (writer): produces a final summary

Without ThriftAI: changing the writer's prompt re-runs all 3 agents.
With ThriftAI: replay researcher + analyzer from trace, only writer goes live.

Run 1 (normal):
    python examples/research_pipeline.py

Run 2 (replay, iterate on writer only):
    python examples/research_pipeline.py --replay --live writer
"""

import argparse
import thriftai as ta


# --- Agent definitions ---

@ta.agent(name="researcher", depends_on=[])
def research(session, topic: str) -> str:
    """Gather raw information about a topic."""
    result = session.completion(
        messages=[
            {"role": "system", "content": "You are a research assistant. Gather key facts about the given topic. Be concise."},
            {"role": "user", "content": f"Research this topic: {topic}"},
        ],
        model="anthropic/claude-sonnet-4-20250514",
    )
    return result


@ta.agent(name="analyzer", depends_on=["researcher"])
def analyze(session, research_output: str) -> str:
    """Analyze the research and identify key insights."""
    result = session.completion(
        messages=[
            {"role": "system", "content": "You are an analyst. Identify the 3 most important insights from the research provided."},
            {"role": "user", "content": f"Analyze this research:\n\n{research_output}"},
        ],
        model="anthropic/claude-sonnet-4-20250514",
    )
    return result


@ta.agent(name="writer", depends_on=["analyzer"])
def write_summary(session, analysis: str) -> str:
    """Write a final summary from the analysis."""
    result = session.completion(
        messages=[
            {"role": "system", "content": "You are a technical writer. Write a clear, engaging summary based on the analysis provided."},
            {"role": "user", "content": f"Write a summary based on this analysis:\n\n{analysis}"},
        ],
        model="anthropic/claude-sonnet-4-20250514",
    )
    return result


# --- Pipeline ---

def run_pipeline(topic: str, replay_trace: str | None = None, live_agents: list[str] | None = None):
    session = ta.Session(cache_dir=".thriftai")

    if replay_trace:
        ctx_manager = session.replay(trace_id=replay_trace, live=live_agents)
    else:
        ctx_manager = session.run()

    with ctx_manager as run:
        research_output = research(run, topic)
        analysis = analyze(run, research_output)
        summary = write_summary(run, analysis)

    print("\n" + "=" * 50)
    print("FINAL SUMMARY:")
    print("=" * 50)
    print(summary)
    print("\n" + run.cost_report.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="The impact of AI on cybersecurity")
    parser.add_argument("--replay", type=str, default=None, help="Trace ID to replay from")
    parser.add_argument("--live", nargs="*", default=None, help="Agents to run live during replay")
    args = parser.parse_args()

    run_pipeline(args.topic, replay_trace=args.replay, live_agents=args.live)
