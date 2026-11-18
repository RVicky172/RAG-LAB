"""
Evaluate the LangGraph agent (Doc 4) with the SAME questions, scores and judge
as plain RAG (Doc 1), so the two compare one-to-one.

The question: does checking the chunks and searching again (when they don't fit)
improve retrieval and answers, and what does it cost in time and tokens?

Run: uv run python -m eval.langgraph_agent.evaluate
Output: eval/langgraph_agent/results/ with results.json and report.html
"""

import asyncio
from pathlib import Path

from app.langgraph_agent.graph import answer
from eval.rag import evaluate as rag_eval

HERE = Path(__file__).parent                  # eval/langgraph_agent/
RESULTS_DIR = HERE / "results"


def main():
    """Run the RAG eval (Doc 1) on the LangGraph agent instead of plain RAG."""
    asyncio.run(rag_eval.main(answer_fn=answer, results_dir=RESULTS_DIR,
                              title="LangGraph agent", project="langgraph-eval"))


if __name__ == "__main__":
    main()
