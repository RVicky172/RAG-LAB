"""
The paper assistant: four ways to answer questions about the 4 AI papers in data/raw/.

    shared/        used by every system: settings, the LLM, the tools, trace printing
    rag/           Doc 1: RAG (ingest the PDFs once, then query them)
    single_agent/  Doc 2: one agent that decides which tools to call
    multi_agent/   Doc 3: a supervisor that delegates to two specialist agents
    langgraph_agent/  Doc 4: a self-correcting RAG agent, built step by step with LangGraph
"""
