# Doc 2: AI agent and its evaluation

**Time:** ~2 hours  **You build:** an agent that decides on its own when to search the papers and when to calculate, plus an eval that checks both *what* it answered and *how* it got there.

**Files you create in this doc:**

| Folder | Files | Step |
|---|---|---|
| `app/shared/` | `tools.py`, `trace.py` | 2 |
| `app/single_agent/` | `agent.py` | 2 |
| `eval/single_agent/` | `diagnose.py`, `report.py`, `evaluate.py` | 3.3 |

Plus an `__init__.py` in each new folder. The question file `eval/single_agent/questions.jsonl` is already there. Everything in `eval/shared/` is reused from Doc 1.

---

## 1. RAG vs agent

Your RAG pipeline is a **fixed recipe**: it always searches once, then answers. It can't handle "search, then do maths on the result", and it searches even for "hi".

An **agent** is an LLM in a loop with **tools**:

```
question ─► LLM: "which tool, if any?" ─► call tool ─► result back to LLM ─┐
                 ▲                                                         │
                 └──────────────── repeat until the LLM writes an answer ◄─┘
```

| Word | Plain meaning |
|---|---|
| **Tool** | A normal Python function the LLM may call. The LLM sees only its **name, arguments and docstring**, so the docstring *is* the instruction. |
| **Tool call** | The LLM outputs "call `search_papers` with `query=...`". The framework runs the function and sends the result back. |
| **Trace** | The full list of messages: question, tool calls, tool results, final answer. You evaluate it. |

**Why LangChain `create_agent`?** It's the standard way to build this loop. One function call gives you the loop, tool calling and the message trace. Your RAG `search()` from Doc 1 becomes a tool, unchanged.

---

## 2. Build the agent (40 min)

The agent gets its own folder, `app/single_agent/`. Its tools, and the function that prints its conversation, go in `app/shared/`, because the multi-agent system in Doc 3 uses them too.

`tools.py` holds the two tools. `search_papers` is your RAG `search()` from Doc 1; `calculator` does safe arithmetic. Read the docstrings: they are what the LLM sees.

**`app/shared/tools.py`**
```python
"""
The tools our agents can call. Both the single agent and the multi-agent system use them.

@tool turns a normal Python function into something the LLM can call.
The LLM sees only the function NAME, its ARGUMENTS and its DOCSTRING, so each
docstring below is written as an instruction for the LLM, not for us.
Changing a docstring changes how the agent behaves.
"""

from langchain.tools import tool

from app.rag.query import build_context, search


@tool
def search_papers(query: str) -> str:
    """Search the research-paper knowledge base (Transformer, RAG, Lost in the Middle, RAGAS).
    Returns the 5 most relevant text chunks with their source file and page."""
    # Reuse the RAG search from Doc 1. A tool must return text, so format the
    # chunks the same way the RAG prompt does: "[file p.N]" + the chunk text.
    return build_context(search(query))


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '28.4 - 26.3' or '512 / 8'."""
    # Safety: the LLM chooses this input, so allow only digits and + - * / ( ) .
    # With everything else rejected, eval() can only do arithmetic.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    return str(eval(expression))
```

`trace.py` prints an agent's conversation one line per message, so you can watch the loop:

**`app/shared/trace.py`**
```python
"""
Print an agent's conversation (its "trace") one line per message.

Both agents use it when you run them from the terminal, so you can see the
loop: question -> tool call -> tool result -> answer. For the full picture
(tokens, timing, what each specialist did inside) open the trace in Phoenix.
"""


def print_trace(messages) -> None:
    """Print one line per message: its tool calls if it made any, else the start of its text.

    Args:
        messages: The `messages` list an agent's `run()` returns.
    """
    for m in messages:
        # Only AI messages have tool_calls, so getattr gives [] for the others.
        calls = [f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", [])]
        print(f"[{m.type}] {calls or m.content[:150]}")
```

The agent's folder:

**`app/single_agent/__init__.py`**
```python
"""
Doc 2: a single agent, one LLM that decides for itself which tools to call.

    agent.py   the agent and its run() function     (python -m app.single_agent.agent "...")

Its tools (search_papers, calculator) are in app/shared/tools.py, because the
multi-agent system uses them too.
"""
```

`agent.py` builds the agent: the LLM, the two tools and a system prompt. `run()` asks one question and returns the answer plus the full trace:

**`app/single_agent/agent.py`**
```python
"""
A single AI agent: one LLM that can use tools (Doc 2).

How it works (the "agent loop"):
    1. The LLM reads the question and decides: call a tool, or answer?
    2. If it calls a tool, LangChain runs the Python function and sends the
       result back to the LLM.
    3. Repeat until the LLM writes a final answer.

Why an agent instead of plain RAG? RAG always searches exactly once. An agent
can search, then calculate with the result, or skip tools entirely for "hi".

Run: uv run python -m app.single_agent.agent "your question"
"""

import sys

from langchain.agents import create_agent

from app.shared import config
from app.shared.llm import get_llm
from app.shared.tools import calculator, search_papers
from app.shared.trace import print_trace
from app.shared.tracing import start_tracing

# The system prompt: rules the LLM follows for every question.
SYSTEM = """You are a research assistant for a small library of AI papers.
Use search_papers to find facts; never answer paper questions from memory.
Use calculator for any arithmetic. Cite sources like [1706.03762v7.pdf p.5].
If the papers don't contain the answer, say so."""

# create_agent builds the whole loop: LLM -> tool call -> tool result -> LLM ...
agent = create_agent(get_llm(), tools=[search_papers, calculator], system_prompt=SYSTEM)


def run(question: str) -> dict:
    """Ask the agent one question.

    Args:
        question: The user's question.

    Returns:
        `{"answer": the final reply, "messages": the full trace}`. The trace is
        every message in order (question, tool calls, tool results, answer).
        The eval needs it to check which tools were used.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    return {"answer": messages[-1].content, "messages": messages}  # last message = final answer


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    start_tracing("single-agent")  # every run shows up in Phoenix, project "single-agent"
    question = " ".join(sys.argv[1:]) or "How much higher is the big Transformer's EN-DE BLEU than the base model's?"
    print_trace(run(question)["messages"])
    print(f"\nFull trace (tokens, timing, every step): {config.PHOENIX_URL} (project single-agent)")
```

```bash
uv run python -m app.single_agent.agent
uv run python -m app.single_agent.agent "What is 512 divided by 8?"
uv run python -m app.single_agent.agent "Hi! What can you help me with?"
```

Read the printed trace: `[human]` → `[ai]` with a tool call → `[tool]` result → `[ai]` final answer. The LLM chose the tool and wrote the search query itself.

Then open the same run in Phoenix (http://localhost:6006 → project `single-agent`): the same loop as a tree, with every LLM call's full prompt, tokens and time, and the `search` step (with its chunks) inside the `search_papers` tool. `agent.py` calls `start_tracing("single-agent")`; nothing else was needed, because LangChain's agent steps are recorded automatically.

> **Safety note:** tools are code the LLM can trigger. Keep them narrow (the calculator rejects anything but arithmetic). Never give an agent a tool that can run arbitrary code or delete things unless a human approves.

---

## 3. Evaluation: outcome *and* behaviour (45 min)

An agent can get the right answer the wrong way, for example by guessing a number from memory instead of searching. So you measure both.

### 3.1 What we measure

| Score | What it checks | How |
|---|---|---|
| `tool_match` | Did it call the right tools? (the set of tools called == the expected set) | plain Python on the trace |
| `goal_accuracy` | Did the conversation reach the goal in the reference? | RAGAS `AgentGoalAccuracyWithReference` (LLM judge reads the whole trace) |
| `steps` | Number of tool calls: cost and loops | count |
| `latency_s` | Time per question | timer |

The test set is `eval/single_agent/questions.jsonl` (already there). It has 8 questions, each with `expected_tools`: search only, calculator only, both, and none (a greeting).

> **Why not RAGAS `ToolCallAccuracy`?** It also compares the tool *arguments* exactly. The agent writes its own search query ("Transformer heads base model"), which never matches a fixed string. Checking which tools were called is the robust part.

### 3.2 Tokens and traces

An agent makes several LLM calls per question (decide → tool → decide → answer), and each one re-sends the prompt, the tool descriptions and the conversation so far. You don't count them yourself: the eval wraps each question in one Phoenix trace (project `single-agent-eval`) that holds every LLM call, tool call and judge call with its tokens. The project page shows the totals of the whole run, and the HTML report links each question to its trace.

### 3.3 The eval files

The eval gets its own folder, `eval/single_agent/`, next to `eval/rag/`. It reuses the judge, the question reader and the report building blocks from `eval/shared/`. Doc 3 reuses this whole folder for the multi-agent system.

The eval folder:

**`eval/single_agent/__init__.py`**
```python
"""
Doc 2: evaluate the single agent. The multi-agent eval (Doc 3) reuses this code.

    questions.jsonl  the test set: question, reference goal, expected tools
    evaluate.py      run the eval; main() also runs the multi-agent eval   (python -m eval.single_agent.evaluate)
    diagnose.py      pass / check / fail per question, and the likely cause
    report.py        write results/report.html (for the multi-agent eval too)
    results/         output of the last run: results.json + report.html
"""
```

`diagnose.py` decides pass / check / fail for each question, following the idea of this doc: **fail** = goal missed, **check** = right answer the wrong way (wrong tools), **pass** = both right. It also says in words what went wrong, e.g. "skipped calculator: it did the arithmetic itself":

**`eval/single_agent/diagnose.py`**
```python
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
```

`report.py` builds the HTML page. Like the other page files, copy it as it is:

<details><summary><b><code>eval/single_agent/report.py</code></b> (click to show)</summary>

**`eval/single_agent/report.py`**
```python
"""
The HTML report for the agent evals (Doc 2 + 3):
eval/single_agent/results/report.html and eval/multi_agent/results/report.html

Page layout, top to bottom:
    1. Summary   passed / check / failed, tool match and goal accuracy, steps and time,
                 and where to find the tokens (Phoenix)
    2. Questions "Needs attention" first, then "Passed". Click a row to see the
                 diagnosis, expected vs actual answer, and expected vs called tools.
                 The full step-by-step trace is one click away, in Phoenix.
"""

from pathlib import Path

from eval.shared.html_page import (esc, page, phoenix_panel, question_list, question_row, stat,
                                   trace_link, verdict)
from eval.shared.status import level
from eval.single_agent.diagnose import diagnose_agent

# The 2 scores: key -> (name, what it checks, a low score points to)
METRICS = {
    "tool_match": ("Tool match", "Did it call exactly the expected tools (or specialists)?",
                   "behaviour: the system prompt or the tool docstrings"),
    "goal_accuracy": ("Goal", "Did the conversation reach the reference goal? (LLM judge reads the trace)",
                      "the answer, or the judge"),
}


def pills(names: list[str]) -> str:
    """Tool names as small labels, or 'none'."""
    if not names:
        return '<span class="muted">none</span>'
    return " ".join(f'<code class="tool">{esc(name)}</code>' for name in names)


def summary(results: list[dict], statuses: list[str], project: str) -> str:
    """Pass/check/fail counts, the two scores, steps and time, and a pointer to Phoenix."""
    count = len(results)

    def mean(key):
        return sum(r[key] for r in results) / count

    score_rows = "".join(
        f'<tr><td><b>{name}</b></td><td class="score {level(mean(key))}">{mean(key):.2f}</td>'
        f'<td>{checks}</td><td class="muted">{points_to}</td></tr>'
        for key, (name, checks, points_to) in METRICS.items())

    return f"""
<div class="stats">
  {stat(statuses.count("pass"), "passed (right tools, goal reached)", "good")}
  {stat(statuses.count("check"), "to check (goal reached the wrong way)", "check")}
  {stat(statuses.count("fail"), "failed (goal missed)", "bad")}
</div>

<h2>Scores</h2>
<div class="panel" style="padding:4px 10px">
<table class="scores">
  <tr><th>Score</th><th>Mean</th><th>What it checks</th><th>If low, look at</th></tr>
  {score_rows}
</table>
</div>
<div class="stats">
  {stat(f"{mean('steps'):.1f}", "tool calls per question")}
  {stat(f"{mean('latency_s'):.1f}s", "per question (one at a time)")}
</div>

<h2>Tokens and traces</h2>
{phoenix_panel(project)}"""


def agent_row(num: int, row: dict) -> tuple[str, str]:
    """One question row of the agent report. Returns (status, html)."""
    diag = diagnose_agent(row)

    cells = (f'<span class="score {level(row["tool_match"])}" data-label="Tools">{"yes" if row["tool_match"] else "no"}</span>'
             f'<span class="score {level(row["goal_accuracy"])}" data-label="Goal">{row["goal_accuracy"]:.2f}</span>'
             f'<span class="num" data-label="Steps">{row["steps"]}</span>'
             f'<span class="num muted" data-label="Time">{row["latency_s"]}s</span>'
             f'<span data-label="Trace">{trace_link(row, "open ↗")}</span>')

    details = f"""
    {verdict(diag)}
    <h3>Answer</h3>
    <div class="compare">
      <div><div class="label">Expected</div><div class="answer-text">{esc(row["reference"])}</div></div>
      <div><div class="label">Got</div><div class="answer-text">{esc(row["answer"])}</div></div>
    </div>
    <h3>Tools</h3>
    <div class="compare">
      <div><div class="label">Expected</div>{pills(row["expected_tools"])}</div>
      <div><div class="label">Called, in order</div>{pills(row["tools_called"])}</div>
    </div>
    <h3>Trace</h3>
    <p>Every step (LLM calls, tool calls with their inputs and results, the judge) with tokens and timing:
       {trace_link(row, "open this question's trace in Phoenix ↗")}</p>"""
    return diag.status, question_row(num, row["question"], diag, cells, details)


def write_agent_report(results: list[dict], path: Path, title: str, project: str) -> None:
    """Write the HTML report for an agent eval run.

    Args:
        results: The rows from `eval.single_agent.evaluate`, each with its scores and `trace_id`.
        path: Where to write the .html file.
        title: Which system was tested, e.g. "Single agent" or "Multi-agent".
        project: Its Phoenix project, e.g. "single-agent-eval".
    """
    rows = [agent_row(num, row) for num, row in enumerate(results, start=1)]
    statuses = [status for status, _ in rows]
    columns = ["Tools", "Goal", "Steps", "Time", "Trace"]

    body = (summary(results, statuses, project)
            + question_list("Needs attention", "Open by default. The coloured box says where the problem "
                            "probably is: the agent's behaviour, its answer, or the judge. It's a hint, not "
                            "a verdict: open the trace to decide.",
                            columns, [html for status, html in rows if status != "pass"])
            + question_list("Passed", "Click a question to see its details.",
                            columns, [html for status, html in rows if status == "pass"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(f"{title} eval report", f"{len(results)} questions · one at a time", body),
                    encoding="utf-8")
```

</details>

`evaluate.py` is the eval: for each question (one trace in Phoenix) it runs the agent, compares the tools called with the expected ones, asks the judge whether the goal was reached, and saves `results.json` and `report.html`. `main()` takes the system to test as an argument, so Doc 3 can reuse it:

**`eval/single_agent/evaluate.py`**
```python
"""
Evaluate the single agent (Doc 2) on HOW it worked (tools) and WHAT it delivered (goal).

For each question in `eval/single_agent/questions.jsonl` we measure:

    tool_match     Did it call exactly the expected set of tools?   (plain code, no LLM)
    goal_accuracy  Did the conversation reach the reference goal?   (LLM judge reads the trace)
    steps          How many tool calls it made (cost, loops)
    latency_s      Seconds for the whole question

Why both? An agent can reach the right answer the wrong way, e.g. doing maths
"in its head" instead of using the calculator. goal_accuracy can't see that.
tool_match can.

Every question is also one trace in Phoenix: every LLM call, tool call and
judge call, with tokens and timing. The report links to each one.

The multi-agent eval (Doc 3) reuses main() below with its own system and names.

Run: uv run python -m eval.single_agent.evaluate
Output: one line per question, the averages, and eval/single_agent/results/
with results.json and report.html (open it in a browser).
"""

import asyncio
import sys
import time
from pathlib import Path

from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.metrics.collections import AgentGoalAccuracyWithReference

from app.shared.tracing import current_trace_id, start_tracing
from app.single_agent.agent import run
from eval.shared.judge import judge_llm
from eval.shared.questions import load_questions, save_results
from eval.single_agent.report import write_agent_report

HERE = Path(__file__).parent                  # eval/single_agent/
QUESTIONS_FILE = HERE / "questions.jsonl"
RESULTS_DIR = HERE / "results"


def tools_called(messages) -> list[str]:
    """List the names of every tool the agent called, in order.

    Args:
        messages: The message trace from `run()`.

    Returns:
        Tool names, e.g. `["search_papers", "calculator"]`. Empty if no tool was used.
    """
    # Only AI messages have tool_calls, so getattr gives [] for the others.
    return [call["name"] for m in messages for call in getattr(m, "tool_calls", [])]


async def main(run_fn=run, questions=QUESTIONS_FILE, results_dir=RESULTS_DIR,
               title="Single agent", project="single-agent-eval"):
    """Run every question through the system, score it, print and save the results.

    The arguments let Doc 3 reuse this same eval for the multi-agent system.

    Args:
        run_fn: The system to test. Takes a question, returns `{"answer", "messages"}`.
        questions: Path to the question file.
        results_dir: Folder for results.json and report.html.
        title: Name of the system, shown in the HTML report.
        project: The Phoenix project the traces go to.
    """
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    tracer = start_tracing(project, judge=True)  # record the agent and the judge in Phoenix
    goal = AgentGoalAccuracyWithReference(llm=judge_llm())
    rows = load_questions(Path(questions))

    # One question at a time, so latency_s is the speed a single user would see.
    for num, row in enumerate(rows, start=1):
        # One trace per question: the agent's calls and the judge's calls all land under it.
        with tracer.start_as_current_span(f"Q{num}: {row['question']}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")   # how Phoenix labels this step
            span.set_attribute("input.value", row["question"])
            row["trace_id"] = current_trace_id()  # so the report can link to it

            # 1. Run the system and time it.
            start = time.perf_counter()
            out = run_fn(row["question"])
            called = tools_called(out["messages"])

            row["answer"] = out["answer"]
            row["tools_called"] = called
            row["steps"] = len(called)
            row["latency_s"] = round(time.perf_counter() - start, 1)
            span.set_attribute("output.value", row["answer"])

            # 2. Behaviour check without an LLM. We compare SETS: order and repeats
            #    don't matter, only "were the right tools used, and no others?"
            row["tool_match"] = float(set(called) == set(row["expected_tools"]))

            # 3. Outcome check: the judge reads the whole trace (converted to RAGAS's
            #    message format) and compares it with the reference goal. Score 0 or 1.
            result = await goal.ascore(
                user_input=convert_to_ragas_messages(out["messages"]),
                reference=row["reference"],
            )
            row["goal_accuracy"] = result.value

        print(f"{row['question'][:50]:50s} tools={called} match={row['tool_match']:.0f} "
              f"goal={row['goal_accuracy']:.0f} {row['latency_s']}s")

    # The averages go in the README results table.
    keys = ["tool_match", "goal_accuracy", "steps", "latency_s"]
    means = [f"{key}={sum(r[key] for r in rows) / len(rows):.2f}" for key in keys]
    print("\nMEAN  " + "  ".join(means))
    print(f"TOKENS  see Phoenix, project {project}: tokens, LLM calls and latency of every question")

    # Save every row so you can read the failing ones later, and the same data as an HTML page.
    results_dir = Path(results_dir)
    save_results(rows, results_dir / "results.json")
    write_agent_report(rows, results_dir / "report.html", title, project)
    print(f"\nreport: {(results_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 3.4 Run it

```bash
uv run python -m eval.single_agent.evaluate   # ~1 min
start eval/single_agent/results/report.html    # open the report (Git Bash on Windows)
```

When this was tested, the means were about `tool_match=0.75–0.88 goal_accuracy=0.75–0.88`. In Phoenix, the `single-agent-eval` project showed about 28,000 tokens for the run (agent + judge), about 2 LLM calls per question. Record yours in the README table.

### 3.5 Read the report

Open `eval/single_agent/results/report.html`:
- **Top:** passed / to check / failed, tool match and goal accuracy with what each one checks, tool calls and seconds per question, and where its traces are in Phoenix.
- **Each question:** a coloured box saying where the problem probably is, the expected answer next to the actual answer, the expected tools next to the tools called, and a link to its **trace** in Phoenix: every step (question → tool call → tool result → answer, then the judge) with tokens and timing.

What the test run revealed:
- On **"How much higher is the big Transformer's BLEU…"** and **"Divide d_model by the number of heads"**, the agent searched but then did the arithmetic **in its head** and skipped `calculator`. The answers were right this time, but mental arithmetic is where LLMs make silent mistakes. A final-answer-only eval would never notice this; `tool_match` did. The report marks these **check**.
- **"What is 512 divided by 8?"** It used the calculator correctly, yet the judge sometimes scores `goal=0`. Open that question in the report, compare the answer with the reference, and decide: agent problem or judge problem?
- **"Hi! What can you help me with?"** No tools needed, none used, but the judge often scores small talk 0. Another judge error to recognise.

---

## 4. Experiment: fix the behaviour (25 min)

Change **one** thing, re-run, compare:

1. Make the system prompt stricter: *"For ANY number you compute, you MUST call calculator, even for simple arithmetic."* Does `tool_match` go up? Does `latency_s`?
2. Or improve the `calculator` docstring instead (the LLM reads it): *"Always use this for arithmetic; never compute in your head."* Which fix works better: prompt or tool description?

---

## Checklist

- [ ] I can explain the agent loop and why the tool docstring matters.
- [ ] I read a trace (in the terminal and in Phoenix) and pointed to the tool call, the tool result and the final answer.
- [ ] I can explain why an agent needs a *behaviour* metric (tools) and not just an *outcome* metric (goal).
- [ ] I changed the prompt or a docstring and measured the effect.

**Next:** [Doc 3: Multi-agent system and its evaluation](03-multi-agent-and-evaluation.md)
