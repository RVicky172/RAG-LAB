"""
Code shared by more than one system.

    config.py   settings, read from .env                    (every system)
    llm.py      the chat model                              (every system)
    tools.py    search_papers + calculator                  (single agent, multi-agent)
    trace.py    print an agent's conversation step by step  (single agent, multi-agent)
    tracing.py  send every step to Arize Phoenix (local)     (every system, every eval)
"""
