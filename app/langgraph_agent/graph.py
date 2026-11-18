"""
Wire the steps from nodes.py into a LangGraph graph, and run it (Doc 4).

    START -> retrieve -> grade --relevant--------------> generate -> END
                ^          | not relevant (one retry)
                +- rewrite +

Why build a graph by hand? create_agent (Docs 2-3) is a ready-made graph where
the LLM decides every step. Here YOU decide the flow, and the LLM only answers
small questions inside it ("are these chunks relevant?"). That makes the
behaviour predictable and easy to test, which is why companies often prefer it
for fixed workflows.

Run: uv run python -m app.langgraph_agent.graph "your question"
"""

import sys

from langgraph.graph import END, START, StateGraph

from app.langgraph_agent.nodes import State, after_grade, generate, grade, retrieve, rewrite
from app.shared import config
from app.shared.tracing import start_tracing


def build_graph():
    """Add the nodes, connect them with edges, and compile the graph.

    Returns:
        A compiled graph: call `.invoke({"question": ...})` on it.
    """
    graph = StateGraph(State)

    # Nodes: a name and the function that runs.
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    # Edges: what runs next.
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])  # after_grade picks one
    graph.add_edge("rewrite", "retrieve")   # the loop: search again with the new query
    graph.add_edge("generate", END)

    return graph.compile()


graph = build_graph()


def answer(question: str) -> dict:
    """Run the graph for one question.

    Same return shape as `app.rag.query.answer()`, so the RAG eval (Doc 1) can
    test both systems with the same questions and scores.

    Args:
        question: The user's question.

    Returns:
        `{"answer", "contexts", "usage"}` like plain RAG, plus `searches`
        (1 or 2) and `query` (the last search query, maybe rewritten).
    """
    state = graph.invoke({"question": question})
    return {
        "answer": state["answer"],
        "contexts": state["chunks"],
        "usage": state["usage"],
        "searches": state["searches"],
        "query": state["query"],
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    start_tracing("langgraph")  # every run shows up in Phoenix, project "langgraph"

    question = " ".join(sys.argv[1:]) or "Which generator model does RAG use?"
    result = answer(question)

    print(result["answer"])
    print(f"\nSearches: {result['searches']}  (last query: {result['query']!r})")
    print("Sources:", [(c["source"], c["page"], c["score"]) for c in result["contexts"]])
    print(f"Trace (each step of the graph): {config.PHOENIX_URL} (project langgraph)")
