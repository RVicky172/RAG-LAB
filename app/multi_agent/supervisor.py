"""
The supervisor of the multi-agent system (Doc 3).

It has no tools of its own: its only "tools" are the two specialists from
specialists.py. It plans, delegates to them, and writes the final answer.

Why split the work? Each agent gets a short prompt and only the tools for its
job, which makes it more reliable. The cost: every delegation is an extra LLM
call (a whole specialist agent runs), so it is slower.

Run: uv run python -m app.multi_agent.supervisor "your question"
"""

import sys

from langchain.agents import create_agent

from app.multi_agent.specialists import ask_analyst, ask_researcher
from app.shared import config
from app.shared.llm import get_llm
from app.shared.trace import print_trace
from app.shared.tracing import start_tracing

supervisor = create_agent(
    get_llm(),
    tools=[ask_researcher, ask_analyst],
    name="supervisor",
    system_prompt=(
        "You manage two specialists. For facts from the papers call ask_researcher. For any "
        "arithmetic call ask_analyst with the numbers. Then write the final answer with citations. "
        "Don't call a specialist for greetings or small talk."
    ),
)


def run(question: str) -> dict:
    """Ask the multi-agent system one question.

    Same return shape as the single agent's `run()`, so the same eval works for both.

    Args:
        question: The user's question.

    Returns:
        `{"answer": the final reply, "messages": the supervisor's trace}`.
        Its tool calls show which specialists it delegated to.
    """
    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    return {"answer": messages[-1].content, "messages": messages}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    start_tracing("multi-agent")  # every run shows up in Phoenix, project "multi-agent"
    question = " ".join(sys.argv[1:]) or "Divide the base Transformer's d_model by its number of attention heads."
    print_trace(run(question)["messages"])
    print(f"\nFull trace, including what each specialist did: {config.PHOENIX_URL} (project multi-agent)")
