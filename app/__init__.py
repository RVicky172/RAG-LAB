"""
The paper assistant: three ways to answer questions about the 4 AI papers in data/raw/.

    shared/        used by every system: settings, the LLM, the tools, trace printing
    rag/           Doc 1: RAG (ingest the PDFs once, then query them)
    single_agent/  Doc 2: one agent that decides which tools to call
    multi_agent/   Doc 3: a supervisor that delegates to two specialist agents
"""
