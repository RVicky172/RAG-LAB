"""
Rebuild the HTML reports from the last saved results, without re-running any eval.
No LLM calls, so it takes a second. Handy when you change the report code.

Run: uv run python -m eval.rebuild_reports
"""

import json

from eval.rag.evaluate import REPORT_FILE, WORKERS
from eval.rag.report import write_rag_report

# The RAG-style evals (Doc 1 + 4): results folder -> title shown in the report, Phoenix project.
for folder, title, project in [("rag", "RAG", "rag-eval"),
                               ("langgraph_agent", "LangGraph agent", "langgraph-eval")]:
    results_file = REPORT_FILE.parents[2] / folder / "results" / "results.json"   # eval/<folder>/results/
    if results_file.exists():
        report_file = results_file.with_name("report.html")
        write_rag_report(json.loads(results_file.read_text(encoding="utf-8")), report_file, WORKERS, title, project)
        print(f"report: {report_file.resolve()}")

# The agent evals (Doc 2 + 3): results folder -> title shown in the report, Phoenix project.
for folder, title, project in [("single_agent", "Single agent", "single-agent-eval"),
                               ("multi_agent", "Multi-agent", "multi-agent-eval")]:
    results_file = REPORT_FILE.parents[2] / folder / "results" / "results.json"   # eval/<folder>/results/
    if not results_file.exists():
        continue  # that eval hasn't been run yet
    # Imported here, not at the top: the agent report is only written in Doc 2.
    from eval.single_agent.report import write_agent_report

    report_file = results_file.with_name("report.html")
    write_agent_report(json.loads(results_file.read_text(encoding="utf-8")), report_file, title, project)
    print(f"report: {report_file.resolve()}")
