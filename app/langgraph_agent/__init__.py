"""
Doc 4: a self-correcting RAG agent, built step by step with LangGraph.

    nodes.py   the shared state, and the steps (nodes): retrieve, grade, rewrite, generate
    graph.py   wires the steps into a graph and runs it    (python -m app.langgraph_agent.graph "...")

    START -> retrieve -> grade --relevant--------------> generate -> END
                ^          | not relevant (one retry)
                +- rewrite +
"""
