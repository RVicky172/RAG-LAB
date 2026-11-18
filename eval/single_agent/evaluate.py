"""
Evaluate the single agent (Doc 2) on HOW it worked (tools) and WHAT it delivered (goal).

For each question in `eval/single_agent/questions.jsonl` we measure:

    tool_match     Did it call exactly the expected set of tools?   (plain code, no LLM)
    goal_accuracy  Did the conversation reach the reference goal?   (LLM judge reads the trace)
    steps          How many tool calls it made (cost, loops)
    latency_s      Seconds for the whole question

Why both? An agent can reach the right answer the wrong way, e.g. doing maths
"in its head" instead of using the calculator. goal_accuracy can't see that.
tool_match can.

Every question is also one trace in Phoenix: every LLM call, tool call and
judge call, with tokens and timing. The report links to each one.

The multi-agent eval (Doc 3) reuses main() below with its own system and names.

Run: uv run python -m eval.single_agent.evaluate
Output: one line per question, the averages, and eval/single_agent/results/
with results.json and report.html (open it in a browser).
"""

import asyncio
import sys
import time
from pathlib import Path

from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.metrics.collections import AgentGoalAccuracyWithReference

from app.shared.tracing import current_trace_id, start_tracing
from app.single_agent.agent import run
from eval.shared.judge import judge_llm
from eval.shared.questions import load_questions, save_results
from eval.single_agent.report import write_agent_report

HERE = Path(__file__).parent                  # eval/single_agent/
QUESTIONS_FILE = HERE / "questions.jsonl"
RESULTS_DIR = HERE / "results"


def tools_called(messages) -> list[str]:
    """List the names of every tool the agent called, in order.

    Args:
        messages: The message trace from `run()`.

    Returns:
        Tool names, e.g. `["search_papers", "calculator"]`. Empty if no tool was used.
    """
    # Only AI messages have tool_calls, so getattr gives [] for the others.
    return [call["name"] for m in messages for call in getattr(m, "tool_calls", [])]


async def main(run_fn=run, questions=QUESTIONS_FILE, results_dir=RESULTS_DIR,
               title="Single agent", project="single-agent-eval"):
    """Run every question through the system, score it, print and save the results.

    The arguments let Doc 3 reuse this same eval for the multi-agent system.

    Args:
        run_fn: The system to test. Takes a question, returns `{"answer", "messages"}`.
        questions: Path to the question file.
        results_dir: Folder for results.json and report.html.
        title: Name of the system, shown in the HTML report.
        project: The Phoenix project the traces go to.
    """
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    tracer = start_tracing(project, judge=True)  # record the agent and the judge in Phoenix
    goal = AgentGoalAccuracyWithReference(llm=judge_llm())
    rows = load_questions(Path(questions))

    # One question at a time, so latency_s is the speed a single user would see.
    for num, row in enumerate(rows, start=1):
        # One trace per question: the agent's calls and the judge's calls all land under it.
        with tracer.start_as_current_span(f"Q{num}: {row['question']}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")   # how Phoenix labels this step
            span.set_attribute("input.value", row["question"])
            row["trace_id"] = current_trace_id()  # so the report can link to it

            # 1. Run the system and time it.
            start = time.perf_counter()
            out = run_fn(row["question"])
            called = tools_called(out["messages"])

            row["answer"] = out["answer"]
            row["tools_called"] = called
            row["steps"] = len(called)
            row["latency_s"] = round(time.perf_counter() - start, 1)
            span.set_attribute("output.value", row["answer"])

            # 2. Behaviour check without an LLM. We compare SETS: order and repeats
            #    don't matter, only "were the right tools used, and no others?"
            row["tool_match"] = float(set(called) == set(row["expected_tools"]))

            # 3. Outcome check: the judge reads the whole trace (converted to RAGAS's
            #    message format) and compares it with the reference goal. Score 0 or 1.
            result = await goal.ascore(
                user_input=convert_to_ragas_messages(out["messages"]),
                reference=row["reference"],
            )
            row["goal_accuracy"] = result.value

        print(f"{row['question'][:50]:50s} tools={called} match={row['tool_match']:.0f} "
              f"goal={row['goal_accuracy']:.0f} {row['latency_s']}s")

    # The averages go in the README results table.
    keys = ["tool_match", "goal_accuracy", "steps", "latency_s"]
    means = [f"{key}={sum(r[key] for r in rows) / len(rows):.2f}" for key in keys]
    print("\nMEAN  " + "  ".join(means))
    print(f"TOKENS  see Phoenix, project {project}: tokens, LLM calls and latency of every question")

    # Save every row so you can read the failing ones later, and the same data as an HTML page.
    results_dir = Path(results_dir)
    save_results(rows, results_dir / "results.json")
    write_agent_report(rows, results_dir / "report.html", title, project)
    print(f"\nreport: {(results_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
