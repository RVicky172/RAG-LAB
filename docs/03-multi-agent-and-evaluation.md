# Doc 3: Multi-agent system and its evaluation

**Time:** ~1.5 hours  **You build:** a supervisor agent that delegates to two specialist agents, then a one-to-one comparison with the single agent from Doc 2.

**Files you create in this doc:**

| Folder | Files | Step |
|---|---|---|
| `app/multi_agent/` | `specialists.py`, `supervisor.py` | 2 |
| `eval/multi_agent/` | `evaluate.py` | 3.2 |

Plus an `__init__.py` in each new folder. The tools and `print_trace` come from `app/shared/` (Doc 2); the eval reuses `eval/single_agent/` (Doc 2).

---

## 1. Why more than one agent?

One agent with many tools gets confused as the tool list and the instructions grow. A common fix is to split the work among **specialists**, each with a short prompt and only the tools for its job, coordinated by a **supervisor**:

```
                      question
                         │
                    SUPERVISOR  (plans, delegates, writes the final answer; no direct tools)
                    ┌────┴─────┐
           ask_researcher    ask_analyst          ← each specialist is wrapped as a tool
                │                 │
           RESEARCHER          ANALYST
          search_papers       calculator
```

| Word | Plain meaning |
|---|---|
| **Supervisor** | An agent whose "tools" are other agents. It decides who works on what. |
| **Specialist** | A normal agent (as in Doc 2) with one job and a narrow toolset. |
| **Delegation / routing** | The supervisor choosing which specialist to call. That choice is what you evaluate. |

**The pattern used here, "agents as tools",** is the simplest correct multi-agent design and the one LangChain recommends. You already know every piece: each specialist is a `create_agent`, wrapped in a `@tool` function.

**Honest trade-off:** every delegation is an extra LLM round-trip, so multi-agent is **slower and costs more**. It's worth it only if it measurably improves quality. That's what this doc tests.

---

## 2. Build it (40 min)

The multi-agent system gets its own folder, `app/multi_agent/`, with one file for the specialists and one for the supervisor. The tools and `print_trace` come from `app/shared/` (Doc 2).

The folder:

**`app/multi_agent/__init__.py`**
```python
"""
Doc 3: a multi-agent system using the supervisor pattern ("agents as tools").

    SUPERVISOR  plans, delegates, writes the final answer (no direct tools)
       |-- ask_researcher -> RESEARCHER agent (tool: search_papers)
       |-- ask_analyst    -> ANALYST agent    (tool: calculator)

    specialists.py   the researcher and the analyst, each wrapped as a tool
    supervisor.py    the supervisor and its run() function   (python -m app.multi_agent.supervisor "...")
"""
```

`specialists.py` builds the two specialists, each a normal agent with one tool, and wraps each one in a `@tool` function so the supervisor can call it:

**`app/multi_agent/specialists.py`**
```python
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
```

`supervisor.py` builds the supervisor. Its only tools are the two specialists. `run()` has the same shape as the single agent's, so the same eval works for both:

**`app/multi_agent/supervisor.py`**
```python
"""
The supervisor of the multi-agent system (Doc 3).

It has no tools of its own: its only "tools" are the two specialists from
specialists.py. It plans, delegates to them, and writes the final answer.

Why split the work? Each agent gets a short prompt and only the tools for its
job, which makes it more reliable. The cost: every delegation is an extra LLM
call (a whole specialist agent runs), so it is slower.

Run: uv run python -m app.multi_agent.supervisor "your question"
"""

import sys

from langchain.agents import create_agent

from app.multi_agent.specialists import ask_analyst, ask_researcher
from app.shared import config
from app.shared.llm import get_llm
from app.shared.trace import print_trace
from app.shared.tracing import start_tracing

supervisor = create_agent(
    get_llm(),
    tools=[ask_researcher, ask_analyst],
    name="supervisor",
    system_prompt=(
        "You manage two specialists. For facts from the papers call ask_researcher. For any "
        "arithmetic call ask_analyst with the numbers. Then write the final answer with citations. "
        "Don't call a specialist for greetings or small talk."
    ),
)


def run(question: str) -> dict:
    """Ask the multi-agent system one question.

    Same return shape as the single agent's `run()`, so the same eval works for both.

    Args:
        question: The user's question.

    Returns:
        `{"answer": the final reply, "messages": the supervisor's trace}`.
        Its tool calls show which specialists it delegated to.
    """
    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    return {"answer": messages[-1].content, "messages": messages}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    start_tracing("multi-agent")  # every run shows up in Phoenix, project "multi-agent"
    question = " ".join(sys.argv[1:]) or "Divide the base Transformer's d_model by its number of attention heads."
    print_trace(run(question)["messages"])
    print(f"\nFull trace, including what each specialist did: {config.PHOENIX_URL} (project multi-agent)")
```

```bash
uv run python -m app.multi_agent.supervisor
uv run python -m app.multi_agent.supervisor "How much higher is the big Transformer's EN-DE BLEU than the base model's?"
```

In the trace, look at the `task` text the supervisor writes to each specialist. That handoff message is where multi-agent systems usually break: missing numbers, a vague task, lost context.

The terminal only shows the supervisor's side. Open the run in Phoenix (project `multi-agent`) to see **inside** each specialist: `ask_analyst` contains the analyst's own agent loop, with its own LLM calls and its `calculator` call. That's where the extra cost of multi-agent comes from.

---

## 3. Evaluation: compare one-to-one with the single agent (30 min)

### 3.1 What changes

Same questions, same scores, same judge, same report as Doc 2. The only difference: the supervisor doesn't call `search_papers` or `calculator` itself, it calls the specialist that owns them. So the expected tools are renamed (`search_papers` → `ask_researcher`, `calculator` → `ask_analyst`), and `tool_match` now measures **routing**: did the supervisor delegate to the right specialists?

### 3.2 The eval file

The eval gets its own folder too. It's short, because it reuses `eval/single_agent/` (the eval loop, the pass / check / fail rules and the report):

The folder:

**`eval/multi_agent/__init__.py`**
```python
"""
Doc 3: evaluate the multi-agent system, reusing the single-agent eval.

    evaluate.py   rename the expected tools to specialists, then run the single-agent eval
                  on the supervisor                         (python -m eval.multi_agent.evaluate)
    results/      output of the last run: results.json + report.html (+ the renamed questions)

The report and its pass / check / fail rules come from eval/single_agent/.
"""
```

`evaluate.py` renames the expected tools, then runs the single-agent eval with the supervisor:

**`eval/multi_agent/evaluate.py`**
```python
"""
Evaluate the multi-agent system (Doc 3) with the SAME questions, metrics and
judge as the single agent (Doc 2), so the two systems compare one-to-one.

The only change: the supervisor doesn't call search_papers or calculator
itself, it calls the specialist that owns them. So we rename each expected
tool to that specialist. tool_match then measures ROUTING: did the supervisor
delegate to the right specialists?

Run: uv run python -m eval.multi_agent.evaluate
Output: eval/multi_agent/results/ with results.json, report.html and the renamed questions
"""

import asyncio
import json
from pathlib import Path

from app.multi_agent.supervisor import run
from eval.shared.questions import load_questions
from eval.single_agent import evaluate as single_agent_eval

HERE = Path(__file__).parent                  # eval/multi_agent/
RESULTS_DIR = HERE / "results"

# Which specialist owns each tool.
TOOL_TO_AGENT = {"search_papers": "ask_researcher", "calculator": "ask_analyst"}


def main():
    """Rename the expected tools, then run the single-agent eval on the multi-agent system."""
    # 1. Load the single agent's questions and rename the expected tools to specialists.
    rows = load_questions(single_agent_eval.QUESTIONS_FILE)
    for row in rows:
        row["expected_tools"] = [TOOL_TO_AGENT[t] for t in row["expected_tools"]]

    # 2. Save them as a new question file (the eval reads questions from a file).
    mapped = RESULTS_DIR / "questions.jsonl"
    mapped.parent.mkdir(parents=True, exist_ok=True)
    mapped.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    # 3. Same eval as Doc 2, but run_fn is the supervisor instead of the single agent.
    asyncio.run(single_agent_eval.main(run_fn=run, questions=mapped, results_dir=RESULTS_DIR,
                                       title="Multi-agent", project="multi-agent-eval"))


if __name__ == "__main__":
    main()
```

### 3.3 Run and compare

```bash
uv run python -m eval.multi_agent.evaluate    # ~1 min
start eval/multi_agent/results/report.html    # same report as Doc 2
```

Open both reports side by side, and put both systems in the README table. Two runs from testing:

| System | Run | tool / routing match | goal accuracy | latency (s) |
|---|---|---|---|---|
| Single agent (Doc 2) | 1 | 0.75 | 0.75 | 1.9 |
| Supervisor + 2 specialists | 1 | 0.88 | 0.88 | 2.9 |
| Single agent (Doc 2) | 2 | 0.88 | 0.88 | 1.8 |
| Supervisor + 2 specialists | 2 | 0.88 | 0.62 | 2.7 |

And the cost, which barely moves between runs. Read it in Phoenix: the projects page lists `single-agent-eval` and `multi-agent-eval` with their total tokens and latency, and each trace shows its LLM calls:

| System | LLM calls per question | agent tokens per question |
|---|---|---|
| Single agent | ~2 | ~1,600 |
| Supervisor + 2 specialists | ~4 | ~2,200 |

**How to read it:**
- **Routing** is usually at least as good as the single agent's tool use: the analyst *always* calculates (its prompt allows nothing else), so the "do maths on a fact" questions improve.
- **Goal accuracy** jumps around (0.88 in one run, 0.62 in the other). In run 2, the judge scored correct answers like "64" as 0: open those questions in the report and you'll see it. With 8 questions, one question is 12 points.
- **The cost is steady:** about 2× the LLM calls, ~1.4× the tokens and ~1.5× the latency, because every delegation runs a whole specialist agent.
- The verdict depends on the use case: for a chat UI the delay may be fine; for a high-volume API it may not be.

### 3.4 What the test run revealed

- On **"Divide d_model by the number of heads"**, the supervisor sent "512 / 8" straight to the analyst **without asking the researcher**. It knew d_model = 512 from training data. Right answer, but no citation and no grounding: on your own private documents, that becomes a hallucination. `tool_match` caught it, and the report marks it **check**: "skipped ask_researcher: it answered from memory without checking the papers".
- With ~8 questions, two runs can differ by 12 points from randomness alone. Real teams use 50–200 questions before trusting a difference.

---

## 4. Experiment (20 min)

Pick one:
1. Add to the supervisor prompt: *"Never use numbers from memory: get every fact from ask_researcher first."* Re-run. Does routing improve? Does latency go up?
2. Run each eval twice. How much do the means move between runs? That's your **noise floor**: smaller differences mean nothing.

---

## Checklist

- [ ] I can draw the supervisor pattern and explain "agents as tools".
- [ ] I can name the cost of multi-agent (LLM calls, tokens, latency) from Phoenix and the reports, and decide when it's worth it.
- [ ] I compared single vs multi-agent on the same questions and metrics.
- [ ] I know why routing (which specialist was called) needs its own metric.

**Next:** [Doc 4: LangGraph agent and its evaluation](04-langgraph-agent-and-evaluation.md)
