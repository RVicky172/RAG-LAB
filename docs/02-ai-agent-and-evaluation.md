# Doc 2: AI agent and its evaluation

**Time:** ~2 hours  **You build:** an agent that decides on its own when to search the papers and when to calculate, plus an eval that checks both *what* it answered and *how* it got there.

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

**`app/agent.py`**
```python
"""A single tool-using agent: the LLM decides which tool to call, reads the result, repeats."""
import sys
from langchain.agents import create_agent
from langchain.tools import tool
from app.llm import get_llm
from app.rag import search

@tool
def search_papers(query: str) -> str:
    """Search the research-paper knowledge base (Transformer, RAG, Lost in the Middle, RAGAS).
    Returns the 5 most relevant text chunks with their source file and page."""
    return "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in search(query))

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '28.4 - 26.3' or '512 / 8'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    return str(eval(expression))   # safe here: input is restricted to arithmetic characters

SYSTEM = """You are a research assistant for a small library of AI papers.
Use search_papers to find facts; never answer paper questions from memory.
Use calculator for any arithmetic. Cite sources like [1706.03762v7.pdf p.5].
If the papers don't contain the answer, say so."""

agent = create_agent(get_llm(), tools=[search_papers, calculator], system_prompt=SYSTEM)

def run(question: str) -> dict:
    """Return the final answer plus the full message trace (needed for evaluation)."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return {"answer": result["messages"][-1].content, "messages": result["messages"]}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    out = run(" ".join(sys.argv[1:]) or "How much higher is the big Transformer's EN-DE BLEU than the base model's?")
    for m in out["messages"]:
        calls = [f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", [])]
        print(f"[{m.type}] {calls or m.content[:150]}")
```

```bash
uv run python -m app.agent
uv run python -m app.agent "What is 512 divided by 8?"
uv run python -m app.agent "Hi! What can you help me with?"
```

Read the printed trace: `[human]` → `[ai]` with a tool call → `[tool]` result → `[ai]` final answer. The LLM chose the tool and wrote the search query itself.

> **Safety note:** tools are code the LLM can trigger. Keep them narrow (the calculator rejects anything but arithmetic). Never give an agent a tool that can run arbitrary code or delete things unless a human approves.

---

## 3. Evaluation: outcome *and* behaviour (45 min)

An agent can get the right answer the wrong way, for example by guessing a number from memory instead of searching. So you measure two things:

| Metric | What it checks | How |
|---|---|---|
| `tool_match` | Did it call the right tools? (the set of tools called == the expected set) | plain Python on the trace |
| `goal_accuracy` | Did the conversation reach the goal in the reference? | RAGAS `AgentGoalAccuracyWithReference` (LLM judge reads the whole trace) |
| `steps` | Number of tool calls: cost and loops | count |
| `latency_s` | Time per question | timer |

The test set is `eval/agent_questions.jsonl`. It has 8 questions, each with `expected_tools`: search only, calculator only, both, and none (a greeting).

> **Why not RAGAS `ToolCallAccuracy`?** It also compares the tool *arguments* exactly. The agent writes its own search query ("Transformer heads base model"), which never matches a fixed string. Checking which tools were called is the robust part.

**`eval/agent_eval.py`**
```python
"""Evaluate the agent on HOW it worked (tools) and WHAT it delivered (goal).

Run: uv run python -m eval.agent_eval
"""
import asyncio, json, sys, time
from pathlib import Path
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.metrics.collections import AgentGoalAccuracyWithReference
from app.agent import run
from eval.rag_eval import judge_llm

def tools_called(messages) -> list[str]:
    return [c["name"] for m in messages for c in getattr(m, "tool_calls", [])]

async def main(run_fn=run, questions="eval/agent_questions.jsonl", out_name="agent_eval"):
    sys.stdout.reconfigure(encoding="utf-8")
    goal = AgentGoalAccuracyWithReference(llm=judge_llm())
    rows = [json.loads(l) for l in Path(questions).read_text(encoding="utf-8").splitlines() if l.strip()]

    for row in rows:
        t = time.perf_counter()
        out = run_fn(row["question"])
        called = tools_called(out["messages"])
        row |= {"answer": out["answer"], "tools_called": called, "steps": len(called),
                "latency_s": round(time.perf_counter() - t, 1),
                "tool_match": float(set(called) == set(row["expected_tools"]))}
        result = await goal.ascore(user_input=convert_to_ragas_messages(out["messages"]), reference=row["reference"])
        row["goal_accuracy"] = result.value
        print(f"{row['question'][:50]:50s} tools={called} match={row['tool_match']:.0f} "
              f"goal={row['goal_accuracy']:.0f} {row['latency_s']}s")

    keys = ["tool_match", "goal_accuracy", "steps", "latency_s"]
    print("\nMEAN  " + "  ".join(f"{k}={sum(r[k] for r in rows) / len(rows):.2f}" for k in keys))
    Path("eval/results").mkdir(exist_ok=True)
    Path(f"eval/results/{out_name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(main())
```

`main()` takes the system to test (`run_fn`) as an argument, so Doc 3 can reuse it for the multi-agent system.

```bash
uv run python -m eval.agent_eval      # ~1 min
```

When this was tested, the result was about `tool_match=0.75 goal_accuracy=0.75`. Record yours in the README table.

### What the test run revealed

- On **"How much higher is the big Transformer's BLEU…"** and **"Divide d_model by the number of heads"**, the agent searched but then did the arithmetic **in its head** and skipped `calculator`. The answers were right this time, but mental arithmetic is where LLMs make silent mistakes. A final-answer-only eval would never notice this; `tool_match` did.
- **"What is 512 divided by 8?"** It used the calculator correctly, yet the judge scored `goal=0`. Open the JSON and decide: agent problem or judge problem?

---

## 4. Experiment: fix the behaviour (25 min)

Change **one** thing, re-run, compare:

1. Make the system prompt stricter: *"For ANY number you compute, you MUST call calculator, even for simple arithmetic."* Does `tool_match` go up? Does `latency_s`?
2. Or improve the `calculator` docstring instead (the LLM reads it): *"Always use this for arithmetic; never compute in your head."* Which fix works better: prompt or tool description?

---

## Checklist

- [ ] I can explain the agent loop and why the tool docstring matters.
- [ ] I read a trace and pointed to the tool call, the tool result and the final answer.
- [ ] I can explain why an agent needs a *behaviour* metric (tools) and not just an *outcome* metric (goal).
- [ ] I changed the prompt or a docstring and measured the effect.

**Next:** [Doc 3: Multi-agent system and its evaluation](03-multi-agent-and-evaluation.md)
