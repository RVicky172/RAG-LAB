# Doc 3: Multi-agent system and its evaluation

**Time:** ~1.5 hours  **You build:** a supervisor agent that delegates to two specialist agents, then a one-to-one comparison with the single agent from Doc 2.

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

**`app/multi_agent.py`**
```python
"""Supervisor pattern: a manager agent delegates to specialist agents, which it calls as tools."""
import sys
from langchain.agents import create_agent
from langchain.tools import tool
from app.agent import calculator, search_papers
from app.llm import get_llm

llm = get_llm()

# --- specialists: each has ONE job and only the tools for that job ---
researcher = create_agent(llm, tools=[search_papers], name="researcher", system_prompt=
    "You find facts in the AI-paper library with search_papers. Return the facts with "
    "citations like [1706.03762v7.pdf p.5]. If the papers don't say, reply 'not found in the papers'.")
analyst = create_agent(llm, tools=[calculator], name="analyst", system_prompt=
    "You do arithmetic. Always use the calculator tool, then state the result in one sentence.")

def _last_text(agent, task: str) -> str:
    return agent.invoke({"messages": [{"role": "user", "content": task}]})["messages"][-1].content

@tool
def ask_researcher(task: str) -> str:
    """Delegate a fact-finding task about the AI papers to the researcher agent."""
    return _last_text(researcher, task)

@tool
def ask_analyst(task: str) -> str:
    """Delegate a calculation to the analyst agent. Include all numbers it needs in the task."""
    return _last_text(analyst, task)

# --- supervisor: plans, delegates, writes the final answer; has no direct tools of its own ---
supervisor = create_agent(llm, tools=[ask_researcher, ask_analyst], name="supervisor", system_prompt=
    "You manage two specialists. For facts from the papers call ask_researcher. For any "
    "arithmetic call ask_analyst with the numbers. Then write the final answer with citations. "
    "Don't call a specialist for greetings or small talk.")

def run(question: str) -> dict:
    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})
    return {"answer": result["messages"][-1].content, "messages": result["messages"]}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    out = run(" ".join(sys.argv[1:]) or "Divide the base Transformer's d_model by its number of attention heads.")
    for m in out["messages"]:
        calls = [f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", [])]
        print(f"[{m.type}] {calls or m.content[:200]}")
```

```bash
uv run python -m app.multi_agent
uv run python -m app.multi_agent "How much higher is the big Transformer's EN-DE BLEU than the base model's?"
```

In the trace, look at the `task` text the supervisor writes to each specialist. That handoff message is where multi-agent systems usually break: missing numbers, a vague task, lost context.

---

## 3. Evaluation: compare one-to-one with the single agent (30 min)

Same questions, same metrics, same judge as Doc 2. The only difference is that the expected tools are renamed to the specialist that owns them. `tool_match` now measures **routing**: did the supervisor delegate to the right specialists?

**`eval/multi_agent_eval.py`**
```python
"""Same questions and metrics as the single agent, so the two systems compare one-to-one.
Expected tools are mapped to the specialist that owns them.

Run: uv run python -m eval.multi_agent_eval
"""
import asyncio, json
from pathlib import Path
from app.multi_agent import run
from eval import agent_eval

TOOL_TO_AGENT = {"search_papers": "ask_researcher", "calculator": "ask_analyst"}

def main():
    rows = [json.loads(l) for l in Path("eval/agent_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["expected_tools"] = [TOOL_TO_AGENT[t] for t in r["expected_tools"]]
    mapped = Path("eval/results/multi_agent_questions.jsonl")
    mapped.parent.mkdir(exist_ok=True)
    mapped.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    asyncio.run(agent_eval.main(run_fn=run, questions=str(mapped), out_name="multi_agent_eval"))

if __name__ == "__main__":
    main()
```

```bash
uv run python -m eval.multi_agent_eval     # ~1 min
```

Put both systems side by side in the README table. From the test run:

| System | tool/routing match | goal accuracy | latency (s) |
|---|---|---|---|
| Single agent (Doc 2) | 0.75 | 0.75 | 1.9 |
| Supervisor + 2 specialists | 0.88 | 0.88 | 2.9 |

**How to read it:** the analyst *always* calculates (its prompt allows nothing else), so the "do maths on a fact" questions improved. The price is ~1.6× latency. The verdict depends on the use case: for a chat UI the delay may be fine; for a high-volume API it may not be.

### What the test run revealed

- On **"Divide d_model by the number of heads"**, the supervisor sent "512 / 8" straight to the analyst **without asking the researcher**. It knew d_model = 512 from training data. Right answer, but no citation and no grounding: on your own private documents, that becomes a hallucination. `tool_match` caught it.
- With ~8 questions, one question is 12 points. Two runs can differ by that much from randomness alone. Real teams use 50–200 questions before trusting a difference.

---

## 4. Experiment (20 min)

Pick one:
1. Add to the supervisor prompt: *"Never use numbers from memory: get every fact from ask_researcher first."* Re-run. Does routing improve? Does latency go up?
2. Run each eval twice. How much do the means move between runs? That's your **noise floor**: smaller differences mean nothing.

---

## Checklist

- [ ] I can draw the supervisor pattern and explain "agents as tools".
- [ ] I can name the cost of multi-agent (latency, tokens) and decide when it's worth it.
- [ ] I compared single vs multi-agent on the same questions and metrics.
- [ ] I know why routing (which specialist was called) needs its own metric.

**Next:** [Doc 4: Deploy on AWS](04-deploy-on-aws.md)
