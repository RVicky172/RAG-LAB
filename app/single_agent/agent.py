"""
A single AI agent: one LLM that can use tools (Doc 2).

How it works (the "agent loop"):
    1. The LLM reads the question and decides: call a tool, or answer?
    2. If it calls a tool, LangChain runs the Python function and sends the
       result back to the LLM.
    3. Repeat until the LLM writes a final answer.

Why an agent instead of plain RAG? RAG always searches exactly once. An agent
can search, then calculate with the result, or skip tools entirely for "hi".

Run: uv run python -m app.single_agent.agent "your question"
"""

import sys

from langchain.agents import create_agent

from app.shared import config
from app.shared.llm import get_llm
from app.shared.tools import calculator, search_papers
from app.shared.trace import print_trace
from app.shared.tracing import start_tracing

# The system prompt: rules the LLM follows for every question.
SYSTEM = """You are a research assistant for a small library of AI papers.
Use search_papers to find facts; never answer paper questions from memory.
Use calculator for any arithmetic. Cite sources like [1706.03762v7.pdf p.5].
If the papers don't contain the answer, say so."""

# create_agent builds the whole loop: LLM -> tool call -> tool result -> LLM ...
agent = create_agent(get_llm(), tools=[search_papers, calculator], system_prompt=SYSTEM)


def run(question: str) -> dict:
    """Ask the agent one question.

    Args:
        question: The user's question.

    Returns:
        `{"answer": the final reply, "messages": the full trace}`. The trace is
        every message in order (question, tool calls, tool results, answer).
        The eval needs it to check which tools were used.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    return {"answer": messages[-1].content, "messages": messages}  # last message = final answer


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    start_tracing("single-agent")  # every run shows up in Phoenix, project "single-agent"
    question = " ".join(sys.argv[1:]) or "How much higher is the big Transformer's EN-DE BLEU than the base model's?"
    print_trace(run(question)["messages"])
    print(f"\nFull trace (tokens, timing, every step): {config.PHOENIX_URL} (project single-agent)")
