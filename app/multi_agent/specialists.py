"""
The two specialist agents of the multi-agent system, each wrapped as a tool.

Each specialist is a normal agent (as in Doc 2) with ONE job and only the tool
for that job. Wrapping it in a @tool function makes it look like any other
tool to the supervisor: the supervisor "calls" ask_researcher, and behind it
a whole agent runs.
"""

from langchain.agents import create_agent
from langchain.tools import tool

from app.shared.llm import get_llm
from app.shared.tools import calculator, search_papers

# One LLM connection shared by both specialists.
llm = get_llm()

researcher = create_agent(
    llm,
    tools=[search_papers],
    name="researcher",
    system_prompt=(
        "You find facts in the AI-paper library with search_papers. Return the facts with "
        "citations like [1706.03762v7.pdf p.5]. If the papers don't say, reply 'not found in the papers'."
    ),
)

analyst = create_agent(
    llm,
    tools=[calculator],
    name="analyst",
    system_prompt="You do arithmetic. Always use the calculator tool, then state the result in one sentence.",
)


def _last_text(agent, task: str) -> str:
    """Run a specialist on one task and return only its final reply.

    The supervisor needs only the result, not the specialist's whole trace.

    Args:
        agent: The specialist agent to run.
        task: The instruction the supervisor wrote for it.

    Returns:
        The text of the specialist's last message.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


# As with any tool, the supervisor LLM reads these docstrings to decide whom to call.

@tool
def ask_researcher(task: str) -> str:
    """Delegate a fact-finding task about the AI papers to the researcher agent."""
    return _last_text(researcher, task)


@tool
def ask_analyst(task: str) -> str:
    """Delegate a calculation to the analyst agent. Include all numbers it needs in the task."""
    # The analyst can't search, so the task text must contain every number it needs.
    return _last_text(analyst, task)
