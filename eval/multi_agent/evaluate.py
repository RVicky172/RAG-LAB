"""
Evaluate the multi-agent system (Doc 3) with the SAME questions, metrics and
judge as the single agent (Doc 2), so the two systems compare one-to-one.

The only change: the supervisor doesn't call search_papers or calculator
itself, it calls the specialist that owns them. So we rename each expected
tool to that specialist. tool_match then measures ROUTING: did the supervisor
delegate to the right specialists?

Run: uv run python -m eval.multi_agent.evaluate
Output: eval/multi_agent/results/ with results.json, report.html and the renamed questions
"""

import asyncio
import json
from pathlib import Path

from app.multi_agent.supervisor import run
from eval.shared.questions import load_questions
from eval.single_agent import evaluate as single_agent_eval

HERE = Path(__file__).parent                  # eval/multi_agent/
RESULTS_DIR = HERE / "results"

# Which specialist owns each tool.
TOOL_TO_AGENT = {"search_papers": "ask_researcher", "calculator": "ask_analyst"}


def main():
    """Rename the expected tools, then run the single-agent eval on the multi-agent system."""
    # 1. Load the single agent's questions and rename the expected tools to specialists.
    rows = load_questions(single_agent_eval.QUESTIONS_FILE)
    for row in rows:
        row["expected_tools"] = [TOOL_TO_AGENT[t] for t in row["expected_tools"]]

    # 2. Save them as a new question file (the eval reads questions from a file).
    mapped = RESULTS_DIR / "questions.jsonl"
    mapped.parent.mkdir(parents=True, exist_ok=True)
    mapped.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    # 3. Same eval as Doc 2, but run_fn is the supervisor instead of the single agent.
    asyncio.run(single_agent_eval.main(run_fn=run, questions=mapped, results_dir=RESULTS_DIR,
                                       title="Multi-agent", project="multi-agent-eval"))


if __name__ == "__main__":
    main()
