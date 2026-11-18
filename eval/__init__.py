"""
The evals: one folder per system, plus the code they share.

    shared/        used by every eval: the judge, question files, the HTML report blocks
    rag/           Doc 1: evaluate the RAG pipeline       (python -m eval.rag.evaluate)
    single_agent/  Doc 2: evaluate the single agent       (python -m eval.single_agent.evaluate)
    multi_agent/   Doc 3: evaluate the multi-agent system (python -m eval.multi_agent.evaluate)
    langgraph_agent/  Doc 4: evaluate the LangGraph agent  (python -m eval.langgraph_agent.evaluate)
    rebuild_reports.py  rebuild every HTML report from the saved results (python -m eval.rebuild_reports)

Each eval folder holds its own questions and writes its own results/ folder:
results.json (every row) and report.html (open it in a browser). Every question is
also a trace in Phoenix (http://localhost:6006), in a project per eval.
"""
