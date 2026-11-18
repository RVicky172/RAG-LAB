"""
Doc 3: a multi-agent system using the supervisor pattern ("agents as tools").

    SUPERVISOR  plans, delegates, writes the final answer (no direct tools)
       |-- ask_researcher -> RESEARCHER agent (tool: search_papers)
       |-- ask_analyst    -> ANALYST agent    (tool: calculator)

    specialists.py   the researcher and the analyst, each wrapped as a tool
    supervisor.py    the supervisor and its run() function   (python -m app.multi_agent.supervisor "...")
"""
