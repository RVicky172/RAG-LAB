"""
HTTP API: the four systems (RAG, agent, multi-agent, LangGraph) behind one web service (Doc 5).

Why? Real apps (a website, a chat UI, another service) call your code over
HTTP, not from a terminal. FastAPI turns plain Python functions into web
endpoints, checks the JSON that comes in, and builds a test page at /docs.

Every system answers through the same endpoint, POST /api/ask/{system}, and
returns the same JSON shape, so the React UI (ui/) can show any of them.
If the UI has been built (ui/dist exists), this server also serves it at /.

Run: uv run uvicorn app.api:api --port 8000   then open http://localhost:8000/docs
"""

import re
import sys
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.langgraph_agent import graph
from app.multi_agent import supervisor
from app.rag import query as rag
from app.shared import config
from app.shared.tracing import current_trace_id, start_tracing
from app.single_agent import agent

# The Windows console uses cp1252 and fails on some characters in the papers.
sys.stdout.reconfigure(encoding="utf-8")

# Every request shows up in Phoenix, project "api".
tracer = start_tracing("api")

api = FastAPI(title="Paper assistant")

# The systems the API knows. Literal makes FastAPI reject any other name (HTTP 422)
# and shows a drop-down with these four in the /docs page.
System = Literal["rag", "agent", "multi-agent", "langgraph"]

# A citation label such as "1706.03762v7.pdf p.5", as written by the RAG prompt and the tools.
SOURCE_LABEL = re.compile(r"[\w.\-]+\.pdf p\.\d+")


class Question(BaseModel):
    """The JSON body the ask endpoint expects: `{"question": "..."}`.

    FastAPI checks incoming requests against this class and returns a clear
    error (HTTP 422) if `question` is missing or not a string.
    """

    question: str


class Step(BaseModel):
    """One thing a system did on the way to its answer, e.g. a search or a tool call."""

    name: str    # e.g. "search", "calculator", "ask_researcher"
    input: str   # what it was called with


class Answer(BaseModel):
    """What every system returns, so the UI can show them all the same way."""

    system: str
    answer: str
    sources: list[str]   # "file p.N" labels the answer is based on
    steps: list[Step]    # what the system did, in order
    seconds: float       # how long the whole answer took
    trace_id: str        # open it in Phoenix: http://localhost:6006/redirects/traces/<trace_id>


# --- One small adapter per system ----------------------------------------------------
# Each system returns a slightly different dict (see its run()/answer()). These
# functions turn each one into the same three things: answer, sources, steps.

def unique(items: list[str]) -> list[str]:
    """Drop duplicates but keep the order.

    Args:
        items: Any list of strings.

    Returns:
        The list without repeats, first occurrence kept.
    """
    return list(dict.fromkeys(items))


def chunk_labels(chunks: list[dict]) -> list[str]:
    """Turn retrieved chunks into "file p.N" labels.

    Args:
        chunks: Chunk dicts from `app.rag.query.search()`.

    Returns:
        One label per page, best match first.
    """
    return unique([f"{c['source']} p.{c['page']}" for c in chunks])


def ask_rag(question: str) -> tuple[str, list[str], list[Step]]:
    """Plain RAG (Doc 1): always exactly one search, then one LLM call.

    Args:
        question: The user's question.

    Returns:
        (answer, sources, steps).
    """
    out = rag.answer(question)
    steps = [Step(name="search", input=question)]
    return out["answer"], chunk_labels(out["contexts"]), steps


def ask_langgraph(question: str) -> tuple[str, list[str], list[Step]]:
    """LangGraph agent (Doc 4): one search, plus a second one with a rewritten query if needed.

    Args:
        question: The user's question.

    Returns:
        (answer, sources, steps).
    """
    out = graph.answer(question)
    steps = [Step(name="search", input=question)]
    if out["searches"] > 1:
        # The grader said "not relevant", so the graph rewrote the query and searched again.
        steps.append(Step(name="rewrite + search", input=out["query"]))
    return out["answer"], chunk_labels(out["contexts"]), steps


def ask_agent_system(run, question: str) -> tuple[str, list[str], list[Step]]:
    """The single agent (Doc 2) or the multi-agent supervisor (Doc 3).

    Both return the same message list, so one function reads both: each tool
    call is a step, and the "file p.N" labels inside the tool results are the sources.

    Args:
        run: `agent.run` or `supervisor.run`.
        question: The user's question.

    Returns:
        (answer, sources, steps).
    """
    out = run(question)
    steps = []
    sources = []
    for m in out["messages"]:
        # AI messages carry the tool calls the LLM decided to make.
        for call in getattr(m, "tool_calls", []):
            # Each tool has one argument (query, expression or task): show its value.
            steps.append(Step(name=call["name"], input=" ".join(str(v) for v in call["args"].values())))
        # Tool messages carry what the tool returned: the chunks or the specialist's reply.
        if m.type == "tool":
            sources += SOURCE_LABEL.findall(m.content)
    return out["answer"], unique(sources), steps


# Which function answers for which system.
SYSTEMS = {
    "rag": ask_rag,
    "agent": lambda q: ask_agent_system(agent.run, q),
    "multi-agent": lambda q: ask_agent_system(supervisor.run, q),
    "langgraph": ask_langgraph,
}


# --- Endpoints ---------------------------------------------------------------------
# These are plain `def` (not `async def`): our code makes blocking calls (Qdrant,
# the LLM), so FastAPI runs each request in a worker thread. Several questions
# can then be answered at the same time.

@api.get("/api/health")
def health():
    """Is the server up, which LLM does it use, and is the knowledge base loaded?

    The UI shows this in its header, so you can see which LLM server answers
    (e.g. localhost, or host.docker.internal from inside the container).

    Returns:
        `{"status": "ok", "llm": model name, "llm_host": host name, "chunks": count}`.
        `chunks` is 0 if Qdrant is empty or not reachable: run the ingest.
    """
    try:
        chunks = rag.client.count(config.COLLECTION).count
    except Exception:
        # No collection yet, or Qdrant is down: the API still runs, it just can't answer.
        chunks = 0
    return {
        "status": "ok",
        "llm": config.LLM_MODEL,
        "llm_host": urlparse(config.LLM_BASE_URL).hostname,  # the host only, never the key
        "chunks": chunks,
    }


@api.post("/api/ask/{system}", response_model=Answer)
def ask(system: System, q: Question):
    """Answer one question with one of the four systems.

    Args:
        system: "rag", "agent", "multi-agent" or "langgraph" (from the URL).
        q: The JSON body, `{"question": "..."}`.

    Returns:
        An `Answer`: the answer, its sources, the steps taken, the time, and the trace id.
    """
    start = time.perf_counter()

    # One span around the whole request, so everything the system does is one trace in Phoenix.
    with tracer.start_as_current_span(f"api {system}") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", q.question)
        trace_id = current_trace_id()
        try:
            answer, sources, steps = SYSTEMS[system](q.question)
        except Exception as e:
            # Usually LM Studio (no model loaded) or Qdrant isn't reachable.
            # 502 = "a server behind me failed", with the reason, so the UI can show it.
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
        span.set_attribute("output.value", answer)

    return Answer(
        system=system,
        answer=answer,
        sources=sources,
        steps=steps,
        seconds=round(time.perf_counter() - start, 1),
        trace_id=trace_id,
    )


# --- The React UI ------------------------------------------------------------------
# After `npm run build` in ui/ (the Dockerfile does this), serve the built page at /.
# Mounted last, so it never hides the /api routes. html=True serves index.html for "/".
UI_DIR = Path(__file__).parent.parent / "ui" / "dist"
if UI_DIR.exists():
    api.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
