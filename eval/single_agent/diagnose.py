"""
Decide, for one agent question, whether it passed and where to look if it didn't.
Used by the single-agent eval and, through it, the multi-agent eval.

Two scores:
    tool_match     ->  behaviour: did it use the right tools (or specialists)?
    goal_accuracy  ->  outcome: did it reach the goal? (LLM judge)

Status: fail = goal missed, check = goal reached the wrong way (wrong tools),
pass = both right. It's a hint for where to look, not a verdict.
"""

from eval.shared.status import BAD, GOOD, Diagnosis


# What it means when an agent skips a tool, by the job the tool does.
# The multi-agent system calls specialists instead of tools, so both names are listed.
SKIPPED = {
    "search": ({"search_papers", "ask_researcher"},
               "it answered from memory without checking the papers, so the answer has no grounding or citation"),
    "maths": ({"calculator", "ask_analyst"},
              "it did the arithmetic itself, where LLMs make silent mistakes"),
}


def tool_problem(expected: set[str], called: set[str]) -> str:
    """Describe in words how the tools called differ from the tools expected.

    Args:
        expected: The tool names the question should use.
        called: The tool names the agent actually used.

    Returns:
        One phrase, e.g. "skipped calculator: it did the arithmetic itself, ...".
    """
    problems = []
    for tool in sorted(expected - called):
        meaning = next((why for names, why in SKIPPED.values() if tool in names), "")
        problems.append(f"skipped {tool}" + (f": {meaning}" if meaning else ""))
    extra = sorted(called - expected)
    if extra and not expected:
        problems.append(f"called {', '.join(extra)} although no tool was needed (wasted calls)")
    elif extra:
        problems.append(f"also called {', '.join(extra)}, which wasn't needed")
    return "; ".join(problems)


def diagnose_agent(row: dict) -> Diagnosis:
    """Decide the status of one agent question and the most likely cause.

    Status: fail if the goal was missed, check if the goal was reached the wrong
    way (wrong tools), pass if both are right.

    Args:
        row: One result row from eval.single_agent.evaluate: expected_tools, tools_called,
            tool_match and goal_accuracy.

    Returns:
        A Diagnosis.
    """
    expected, called = set(row["expected_tools"]), set(row["tools_called"])
    tools_ok = row["tool_match"] == 1
    goal = row["goal_accuracy"]
    goal_ok = goal >= GOOD
    goal_status = "check" if goal >= BAD else "fail"

    if tools_ok and goal_ok:
        return Diagnosis("pass")
    if goal_ok:
        return Diagnosis("check", "Behaviour", f"Right answer, wrong way: {tool_problem(expected, called)}.")
    if tools_ok and not expected:
        return Diagnosis(goal_status, "Judge?",
                         "No tools were needed and none were used. Judges often score small talk 0: "
                         "read the answer yourself.")
    if tools_ok:
        return Diagnosis(goal_status, "Answer or judge",
                         "It used the right tools, but the judge says the goal wasn't reached. "
                         "Compare the answer with the reference: LLM judges are noisy.")
    return Diagnosis(goal_status, "Agent",
                     f"Goal missed and wrong tools: {tool_problem(expected, called)}.")
