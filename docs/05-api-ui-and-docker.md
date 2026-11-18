# Doc 5: API, UI and Docker

**Time:** ~2 hours  **You build:** one HTTP API in front of all four systems (RAG, agent, multi-agent, LangGraph), a React page to talk to them, and one container that holds both, running next to Qdrant and Phoenix in Docker Compose. Everything runs on your PC with LM Studio.

**Files you create in this doc:**

| Where | Files | Step |
|---|---|---|
| `app/` | `api.py` | 2 |
| `ui/` | `package.json` (by npm), `vite.config.js`, `index.html` | 3 |
| `ui/src/` | `main.jsx`, `systems.js`, `api.js`, `App.jsx`, `AnswerCard.jsx`, `styles.css` | 3 |
| repo root | `Dockerfile`, `.dockerignore`, plus an `app` service in `docker-compose.yml` | 4 |

No Python code from Docs 1-4 changes. The API only imports it.

**Before you start:** Docs 1-4 done, Qdrant filled (`uv run python -m app.rag.ingest`), LM Studio running with the model loaded, and [Node.js](https://nodejs.org) LTS installed (check with `node -v`).

---

## 1. What we add

```
DEVELOPMENT (step 3)                          CONTAINER (step 4)
────────────────────                          ─────────────────────────────────────────────
browser ──► React UI (:5173, npm run dev)     browser ──► :8000
               │ /api/...                                   │        docker compose
               ▼                                ┌───────────▼──────────────────────────────┐
            FastAPI (:8000) ──► Qdrant           │ app: FastAPI + built UI ──► qdrant       │
               │                ──► Phoenix      │                        ──► phoenix       │
               ▼                                 └───┬──────────────────────────────────────┘
            LM Studio (your GPU)                     ▼
                                                 LM Studio (your GPU, host.docker.internal)
```

- **One API for four systems.** Real apps call your code over HTTP, not from a terminal. Every system answers through the same endpoint and returns the same JSON, so one UI can show them all.
- **A UI to see the difference.** The page shows each answer with the *steps* the system took (searches, tool calls, delegations) and a link to the full trace. "Compare all" sends one question to all four systems side by side.
- **One container.** Your code, Python, every library and the built UI, frozen into one image. That image is what a company would deploy; section 6 says where.

---

## 2. The API (30 min)

**FastAPI** turns plain Python functions into web endpoints. Three endpoints are enough:

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Is the server up, which LLM does it use, how many chunks are in Qdrant? |
| `POST /api/ask/{system}` | Answer `{"question": "..."}` with `rag`, `agent`, `multi-agent` or `langgraph`. |
| `GET /docs` | FastAPI's built-in test page (free). |

The four systems return slightly different things (`answer()` in Docs 1 and 4 returns chunks; `run()` in Docs 2 and 3 returns messages). A small **adapter** per system turns each into the same shape: `answer`, `sources`, `steps`, `seconds`, `trace_id`. That's the main idea of this file: the UI never needs to know how a system works inside.

**`app/api.py`**
```python
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
```

Things worth noticing:
- **`Literal[...]`** for `system`: FastAPI rejects `/api/ask/foo` with a clear 422 error, and `/docs` shows a drop-down.
- **`response_model=Answer`**: FastAPI checks your *output* too, and documents it in `/docs`.
- **`def`, not `async def`**: our code blocks while it waits for the LLM. FastAPI runs plain `def` endpoints in worker threads, so the server stays responsive.
- **Steps from messages**: the agents' tool calls are already in their message list (Doc 2's `print_trace` printed the same thing). The API just turns them into JSON.
- **Tracing**: each request is one trace in Phoenix, project `api`. The `trace_id` goes back to the UI so it can link to it.

Run it (LM Studio, Qdrant and Phoenix running as in Docs 1-4):

```bash
uv run uvicorn app.api:api --port 8000
```

In a second terminal:

```bash
curl -s localhost:8000/api/health
# {"status":"ok","llm":"openai/gpt-oss-20b","llm_host":"localhost","chunks":304}

curl -s -X POST localhost:8000/api/ask/agent -H "Content-Type: application/json" \
  -d '{"question": "Divide the base Transformer d_model by its number of attention heads."}'
# {"system":"agent","answer":"... 512 / 8 = 64 ...","sources":["1706.03762v7.pdf p.5", ...],
#  "steps":[{"name":"search_papers","input":"Transformer d_model number of attention heads"},
#           {"name":"calculator","input":"512 / 8"}],"seconds":2.7,"trace_id":"a662..."}
```

Or open **http://localhost:8000/docs**, pick `POST /api/ask/{system}` → *Try it out*. Try `foo` as the system to see the 422 error.

---

## 3. The React UI (45 min)

**What you need:** [Node.js](https://nodejs.org) LTS (it includes `npm`). Check with `node -v`.

**Why React + Vite?** React is what most company front-ends use: the page is built from *components* (functions that return HTML-like JSX) that re-draw themselves when their *state* changes. **Vite** is the dev server and build tool: instant reload while you work, and one command to produce plain HTML/JS/CSS for production.

| Word | Plain meaning |
|---|---|
| **Component** | A function that returns what to draw, e.g. `AnswerCard`. |
| **Props** | The inputs a component gets: `<AnswerCard systemId="rag" result={...} />`. |
| **State** (`useState`) | A value that, when you change it, makes React re-draw the page, e.g. the list of questions asked. |
| **Effect** (`useEffect`) | Code that runs after drawing, e.g. "call `/api/health` once when the page opens". |

### 3.1 Create the project

```bash
mkdir -p ui/src && cd ui
npm init -y
npm install react react-dom
npm install -D vite @vitejs/plugin-react
cd ..
```

Then edit `ui/package.json` so its `"type"` and `"scripts"` look like this (keep the versions npm wrote; this doc was tested with React 19.3, Vite 8.3, plugin-react 6.3):

```json
{
  "name": "paper-assistant-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": { "react": "^19.3.1", "react-dom": "^19.3.1" },
  "devDependencies": { "@vitejs/plugin-react": "^6.3.0", "vite": "^8.3.8" }
}
```

Add to `.gitignore` (they're regenerated by `npm install` and `npm run build`):
```
ui/node_modules/
ui/dist/
```
Do commit `ui/package-lock.json`: it pins exact versions, and the Dockerfile installs from it.

### 3.2 The files

**`ui/vite.config.js`**: the proxy is why the UI can always call `/api/...`, in development and in production.
```js
/**
 * Vite settings for the React UI (Doc 5).
 *
 * What: Vite is the dev server and build tool. `npm run dev` serves the UI at
 * http://localhost:5173 and reloads it on every save; `npm run build` writes
 * the finished page to ui/dist/, which FastAPI then serves.
 *
 * Why the proxy: in development the UI (port 5173) and the API (port 8000) are
 * two servers. The proxy forwards every /api request to FastAPI, so the code
 * can always call "/api/..." - the same URL it uses when FastAPI serves the
 * built UI itself. No CORS setup needed.
 */

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

**`ui/index.html`**: the one HTML page. React fills in the `#root` div.
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Paper assistant</title>
  </head>
  <body>
    <!-- React draws the whole page inside this div (see src/main.jsx). -->
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

**`ui/src/main.jsx`**
```jsx
/**
 * Entry point of the UI: draw the <App> component into the #root div of index.html.
 *
 * StrictMode only runs extra checks in development (it warns about common
 * React mistakes); it changes nothing in the built page.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

**`ui/src/systems.js`**: the four systems and the example questions, in one place.
```js
/**
 * The four systems the UI can talk to, with a one-line reminder of how each works.
 *
 * The `id` is the {system} part of the URL POST /api/ask/{system} in app/api.py.
 */

export const SYSTEMS = [
  { id: "rag", name: "RAG", doc: "Doc 1", how: "Always searches once, then answers from the chunks." },
  { id: "agent", name: "Agent", doc: "Doc 2", how: "One LLM that decides when to search or calculate." },
  { id: "multi-agent", name: "Multi-agent", doc: "Doc 3", how: "A supervisor that delegates to a researcher and an analyst." },
  { id: "langgraph", name: "LangGraph", doc: "Doc 4", how: "Searches, grades the chunks, rewrites the query once if needed." },
];

// Questions to click, one per kind of task: a fact, arithmetic, a comparison, small talk.
export const EXAMPLES = [
  "How many attention heads does the base Transformer use?",
  "How much higher is the big Transformer EN-DE BLEU than the base model?",
  "Which two tasks does the Lost in the Middle paper use?",
  "What does RAGAS measure with faithfulness?",
  "Hi! What can you help me with?",
];

// Where Phoenix shows a trace. Docker publishes it on this port, so it also works with the container.
export const PHOENIX_URL = "http://localhost:6006";
```

**`ui/src/api.js`**: the only file that calls the server.
```js
/**
 * The only file that talks to the FastAPI server (app/api.py).
 *
 * Why keep it separate: the components then only deal with data, and if an
 * endpoint changes there is one place to fix.
 *
 * All URLs start with /api: in development Vite forwards them to
 * localhost:8000 (vite.config.js); in production FastAPI serves this page
 * itself, so they go to the same server.
 */

/**
 * Ask the server whether it is up, which LLM it uses and how many chunks Qdrant holds.
 *
 * @returns {Promise<{status: string, llm: string, llm_host: string, chunks: number}>}
 */
export async function getHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Ask one system one question.
 *
 * @param {string} system - "rag", "agent", "multi-agent" or "langgraph".
 * @param {string} question - The user's question.
 * @returns {Promise<{system: string, answer: string, sources: string[],
 *   steps: {name: string, input: string}[], seconds: number, trace_id: string}>}
 * @throws {Error} With the server's reason, e.g. when LM Studio isn't running.
 */
export async function ask(system, question) {
  const res = await fetch(`/api/ask/${system}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    // FastAPI puts the reason in "detail". It's a string for our own errors (502)
    // and a list for validation errors (422), so fall back to the status code.
    const body = await res.json().catch(() => ({}));
    const reason = typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
    throw new Error(reason);
  }
  return res.json();
}
```

**`ui/src/App.jsx`**: the page. It holds the state (selected system, question, the list of turns) and sends the questions.
```jsx
/**
 * The whole page: pick a system (or compare all four), ask a question, see the answers.
 *
 * Every question becomes a "turn": the question plus one result per system
 * asked. Each system's answer fills in as soon as it arrives, so in "Compare
 * all" mode you also see which system is fastest.
 *
 * Each question is answered on its own: the systems have no chat memory
 * (Docs 1-4 built them that way), so a follow-up like "and the big one?" won't work.
 */

import { useEffect, useState } from "react";

import AnswerCard from "./AnswerCard.jsx";
import { ask, getHealth } from "./api.js";
import { EXAMPLES, SYSTEMS } from "./systems.js";

const COMPARE = "compare";

/**
 * The page component.
 *
 * @returns {JSX.Element}
 */
export default function App() {
  const [health, setHealth] = useState(null); // null = still checking, "down" = no answer
  const [mode, setMode] = useState("rag");    // a system id, or COMPARE for all four
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]);     // newest first

  // Check the server once when the page opens.
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth("down"));
  }, []);

  // Don't send a new question while one is still being answered: locally there
  // is one GPU, and parallel questions would only slow each other down.
  const busy = turns.some((turn) => Object.values(turn.results).some((r) => r.status === "loading"));

  /**
   * Store one system's result in one turn. Uses the "previous state" form of
   * setTurns because several answers can arrive at about the same time.
   *
   * @param {number} turnId - Which turn.
   * @param {string} systemId - Which system's result.
   * @param {object} result - `{status, data}` or `{status, error}`.
   */
  function setResult(turnId, systemId, result) {
    setTurns((prev) =>
      prev.map((turn) =>
        turn.id === turnId ? { ...turn, results: { ...turn.results, [systemId]: result } } : turn,
      ),
    );
  }

  /**
   * Send the question to the chosen system(s) and add a new turn for it.
   *
   * @param {string} text - The question.
   */
  function submit(text) {
    const q = text.trim();
    if (!q || busy) {
      return;
    }

    const systemIds = mode === COMPARE ? SYSTEMS.map((s) => s.id) : [mode];
    const turnId = Date.now();

    // Start every result as "loading", then fill each one in when its answer arrives.
    const results = Object.fromEntries(systemIds.map((id) => [id, { status: "loading" }]));
    setTurns((prev) => [{ id: turnId, question: q, systemIds, results }, ...prev]);
    setQuestion("");

    for (const id of systemIds) {
      ask(id, q)
        .then((data) => setResult(turnId, id, { status: "done", data }))
        .catch((err) => setResult(turnId, id, { status: "error", error: err.message }));
    }
  }

  /**
   * Enter sends, Shift+Enter makes a new line (like most chat apps).
   *
   * @param {KeyboardEvent} e
   */
  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(question);
    }
  }

  const selected = SYSTEMS.find((s) => s.id === mode);

  return (
    <div className="page">
      <header className="top">
        <h1>Paper assistant</h1>
        <Status health={health} />
      </header>

      <section className="panel">
        <div className="modes" role="tablist">
          {SYSTEMS.map((s) => (
            <button key={s.id} role="tab" aria-selected={mode === s.id} onClick={() => setMode(s.id)}>
              {s.name}
            </button>
          ))}
          <button role="tab" aria-selected={mode === COMPARE} onClick={() => setMode(COMPARE)}>
            Compare all
          </button>
        </div>
        <p className="muted how">
          {selected ? selected.how : "Sends the question to all four systems and shows the answers side by side."}
        </p>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(question);
          }}
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about the Transformer, RAG, Lost in the Middle or RAGAS papers…"
            rows={2}
          />
          <button type="submit" className="primary" disabled={busy || !question.trim()}>
            {busy ? "Answering…" : "Ask"}
          </button>
        </form>

        <div className="chips examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => submit(ex)} disabled={busy}>
              {ex}
            </button>
          ))}
        </div>
      </section>

      {turns.map((turn) => (
        <section key={turn.id} className="turn">
          <p className="question">{turn.question}</p>
          <div className={turn.systemIds.length > 1 ? "grid" : ""}>
            {turn.systemIds.map((id) => (
              <AnswerCard key={id} systemId={id} result={turn.results[id]} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

/**
 * The status line in the header: which LLM answers and how many chunks Qdrant holds.
 *
 * @param {object} props
 * @param {object|string|null} props.health - The /api/health JSON, "down", or null while checking.
 * @returns {JSX.Element}
 */
function Status({ health }) {
  if (health === null) {
    return <span className="status muted">checking…</span>;
  }
  if (health === "down") {
    return <span className="status bad">API not reachable: is uvicorn running on port 8000?</span>;
  }
  if (health.chunks === 0) {
    return <span className="status bad">Qdrant is empty: run the ingest</span>;
  }
  return (
    <span className="status ok">
      {health.llm} on {health.llm_host} · {health.chunks} chunks
    </span>
  );
}
```

**`ui/src/AnswerCard.jsx`**: one system's answer.
```jsx
/**
 * One system's answer to one question: the answer text, the steps it took,
 * its sources, the time, and a link to the full trace in Phoenix.
 *
 * Showing the steps next to the answer is the point of this UI: you can see
 * that RAG always searches once, that the agent sometimes skips the search,
 * and that the multi-agent supervisor delegates instead of searching.
 */

import { PHOENIX_URL, SYSTEMS } from "./systems.js";

// When tracing is off (TRACING=false) the API returns an all-zero trace id.
const NO_TRACE = "0".repeat(32);

/**
 * Draw one answer card.
 *
 * @param {object} props
 * @param {string} props.systemId - Which system answered, e.g. "rag".
 * @param {{status: "loading"|"done"|"error", data?: object, error?: string}} props.result
 * @returns {JSX.Element}
 */
export default function AnswerCard({ systemId, result }) {
  const system = SYSTEMS.find((s) => s.id === systemId);

  return (
    <article className="card">
      <header className="card-head">
        <span className="card-title">{system.name}</span>
        <span className="muted">{system.doc}</span>
        {result.status === "done" && <span className="time">{result.data.seconds} s</span>}
      </header>

      {result.status === "loading" && <p className="muted thinking">Thinking…</p>}

      {result.status === "error" && (
        <p className="error">
          {result.error}
          <br />
          <span className="muted">Is LM Studio running with the model loaded, and Qdrant up?</span>
        </p>
      )}

      {result.status === "done" && <Details data={result.data} />}
    </article>
  );
}

/**
 * Show the model's **bold** markdown as bold text.
 *
 * gpt-oss writes a little markdown, mostly **bold**. A full markdown library
 * would be one more dependency, so we handle only this one case: splitting
 * on "**" makes every odd-numbered piece the bold part.
 *
 * @param {string} text - The answer text.
 * @returns {Array<string|JSX.Element>} Plain strings and <strong> elements, in order.
 */
function withBold(text) {
  return text.split("**").map((piece, i) => (i % 2 === 1 ? <strong key={i}>{piece}</strong> : piece));
}

/**
 * The body of a finished card.
 *
 * @param {object} props
 * @param {object} props.data - The JSON the API returned (see app/api.py, class Answer).
 * @returns {JSX.Element}
 */
function Details({ data }) {
  const hasTrace = data.trace_id !== NO_TRACE;

  return (
    <>
      {/* pre-wrap keeps the model's line breaks; the answer is plain text, not HTML. */}
      <p className="answer">{withBold(data.answer)}</p>

      <h4>Steps ({data.steps.length})</h4>
      {data.steps.length === 0 ? (
        <p className="muted">No tools used: answered directly.</p>
      ) : (
        <ol className="steps">
          {data.steps.map((step, i) => (
            <li key={i}>
              <code>{step.name}</code> {step.input}
            </li>
          ))}
        </ol>
      )}

      {data.sources.length > 0 && (
        <>
          <h4>Sources</h4>
          <div className="chips">
            {data.sources.map((source) => (
              <span key={source} className="chip">{source}</span>
            ))}
          </div>
        </>
      )}

      {hasTrace && (
        <a className="trace" href={`${PHOENIX_URL}/redirects/traces/${data.trace_id}`} target="_blank" rel="noreferrer">
          Open trace in Phoenix (tokens, timing, every LLM call) →
        </a>
      )}
    </>
  );
}
```

**`ui/src/styles.css`**: plain CSS with light and dark mode. Nothing here to learn for this project, so copy it as is.
<details><summary>Show styles.css</summary>

```css
/* Styles for the whole UI. Plain CSS, no framework: colours are variables,
   so dark mode only has to change the variables. */

:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --text: #1c1f24;
  --muted: #6b7280;
  --border: #e3e6ea;
  --accent: #2f6fde;
  --accent-text: #ffffff;
  --ok: #1f8a4c;
  --bad: #c2410c;
  --code-bg: #eef1f5;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121418;
    --surface: #1b1e24;
    --text: #e6e8eb;
    --muted: #9aa1ab;
    --border: #2c3038;
    --accent: #5b8ff0;
    --ok: #4ec27f;
    --bad: #f0884f;
    --code-bg: #262a32;
  }
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
}

.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 16px 64px;
}

.top {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 16px;
}

h1 {
  margin: 0;
  font-size: 1.5rem;
}

h4 {
  margin: 14px 0 6px;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
}

.muted {
  color: var(--muted);
}

.status {
  font-size: 0.875rem;
}

.status.ok::before {
  content: "● ";
  color: var(--ok);
}

.status.bad {
  color: var(--bad);
}

/* --- The question panel ------------------------------------------------------ */

.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
}

.modes {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.modes button {
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text);
  padding: 6px 14px;
  border-radius: 999px;
  cursor: pointer;
  font: inherit;
}

.modes button[aria-selected="true"] {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-text);
}

.how {
  margin: 10px 0 12px;
  font-size: 0.9rem;
}

form {
  display: flex;
  gap: 8px;
}

textarea {
  flex: 1;
  resize: vertical;
  font: inherit;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
}

button.primary {
  border: none;
  border-radius: 8px;
  padding: 0 20px;
  background: var(--accent);
  color: var(--accent-text);
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}

button:disabled {
  opacity: 0.5;
  cursor: default;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.examples {
  margin-top: 12px;
}

.chip {
  border: 1px solid var(--border);
  background: var(--code-bg);
  color: var(--text);
  border-radius: 6px;
  padding: 3px 8px;
  font: inherit;
  font-size: 0.8rem;
}

button.chip {
  cursor: pointer;
}

/* --- Answers --------------------------------------------------------------------- */

.turn {
  margin-top: 28px;
}

.question {
  font-weight: 600;
  margin: 0 0 10px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  min-width: 0; /* lets long words wrap inside a grid cell */
}

.card-head {
  display: flex;
  gap: 8px;
  align-items: baseline;
  font-size: 0.875rem;
}

.card-title {
  font-weight: 700;
}

.time {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}

.answer {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.5;
}

.thinking {
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  50% {
    opacity: 0.4;
  }
}

.error {
  color: var(--bad);
  overflow-wrap: anywhere;
}

.steps {
  margin: 0;
  padding-left: 20px;
  font-size: 0.875rem;
  overflow-wrap: anywhere;
}

.steps code {
  background: var(--code-bg);
  border-radius: 4px;
  padding: 1px 5px;
}

.trace {
  display: inline-block;
  margin-top: 14px;
  font-size: 0.85rem;
  color: var(--accent);
}

@media (max-width: 560px) {
  form {
    flex-direction: column;
  }

  button.primary {
    padding: 10px;
  }
}
```
</details>

### 3.3 Run it (development mode)

**First, check what must already be running.** The UI only shows what the API returns, and the API needs the same things as Docs 1-4:

| Needs | Start it with | Check |
|---|---|---|
| Docker Desktop, then Qdrant + Phoenix | `docker compose up -d` | http://localhost:6333/dashboard |
| LM Studio with the model loaded | `lms server start` then `lms load openai/gpt-oss-20b --context-length 32768 -y` | `lms ps` lists the model |
| Chunks in Qdrant | `uv run python -m app.rag.ingest` (once; they survive restarts) | prints `stored: ...` |
| UI packages | `cd ui && npm install` (once, and after `package.json` changes) | a `ui/node_modules/` folder exists |

Then two terminals: the API from step 2, and the UI dev server.

```bash
uv run uvicorn app.api:api --port 8000      # terminal 1 (repo root)
cd ui && npm run dev                         # terminal 2 -> open http://localhost:5173
```

Both keep running until you press **Ctrl+C** in their terminal. In this mode Vite re-draws the page every time you save a file in `ui/src/`. Change a Python file and you must restart uvicorn (or start it with `--reload`, which restarts it on every save).

The header shows `● openai/gpt-oss-20b on localhost · 304 chunks` (your chunk count depends on your chunk size). Then try:

1. **Each system on its own** with the examples. Watch the *Steps*: RAG always does one `search`; the agent picks `search_papers` and maybe `calculator`; the multi-agent supervisor calls `ask_researcher` / `ask_analyst`; LangGraph sometimes shows `rewrite + search`.
2. **"Hi! What can you help me with?"** with Agent and RAG. The agent answers with no steps; RAG searches anyway (Doc 2's whole point, now visible).
3. **Compare all** with the BLEU question: four cards fill in as their answers arrive. Compare steps and seconds.
4. **Open trace in Phoenix** on any card: every LLM call with tokens and timing.
5. **Break something:** unload the model (`lms unload --all`) and ask again. The card shows the 502 reason instead of a blank page. Load it back afterwards.

> **Why no chat memory?** Every question is answered on its own, because Docs 1-4 built the systems that way (and the evals test single questions). A follow-up like "and the big model?" won't work. Adding memory means sending the earlier messages along: a good next exercise, not needed here.

### 3.4 One server for both (production mode)

```bash
cd ui && npm run build && cd ..             # writes ui/dist/ (plain HTML, JS, CSS)
uv run uvicorn app.api:api --port 8000      # now http://localhost:8000 IS the UI
```

`app/api.py` serves `ui/dist` at `/` when it exists. In production there's no Vite and no Node: one Python server answers both the page and the API, on one port. That's what goes into the container.

Two things to remember in this mode:
- `api.py` looks for `ui/dist` **when uvicorn starts**. Build first, then start (or restart) uvicorn.
- The page is a snapshot: after changing a file in `ui/src/`, run `npm run build` again and reload the browser.

### 3.5 When something doesn't work

The UI tells you what's wrong in the header or on the answer card:

| You see | Why | Fix |
|---|---|---|
| Header: *API not reachable* | uvicorn isn't running on port 8000 | start it (terminal 1) |
| Header: *Qdrant is empty: run the ingest* | no chunks (or Qdrant is down) | `docker compose up -d`, then `uv run python -m app.rag.ingest` |
| Card: *APIConnectionError: Connection error.* | LM Studio isn't running | `lms server start` |
| Card: an error that mentions the model | the server runs but no model is loaded | `lms load openai/gpt-oss-20b --context-length 32768 -y` |
| `http://localhost:8000` shows `{"detail":"Not Found"}` | no `ui/dist` when uvicorn started | `npm run build` in `ui/`, then restart uvicorn |
| uvicorn: *address already in use* | something else uses port 8000, often the app container from step 4 | `docker compose --profile app down`, or use `--port 8001` (and change the port in `vite.config.js`) |
| `npm: command not found` | Node.js isn't installed | install Node.js LTS, open a new terminal |
| The *Open trace* link doesn't load | Phoenix isn't running | `docker compose up -d` |

---

## 4. Package it as a container (20 min)

A container image = your code + Python + every library, frozen. It runs the same on any machine with Docker, which is why companies ship containers.

**`Dockerfile`**: two stages. Node is needed only to *build* the UI, so it stays out of the final image.
```dockerfile
# The app image: FastAPI + the four systems + the built React UI (Doc 5).
# Build and run: docker compose --profile app up -d --build

# --- Stage 1: build the React UI --------------------------------------------------
# Node is only needed to BUILD the UI (ui/ -> ui/dist, plain HTML/JS/CSS).
# The final image doesn't need Node at all, so it is a separate stage.
FROM node:24-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui ./
RUN npm run build

# --- Stage 2: the Python app ------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /app

# Install the libraries first: this layer is cached until uv.lock changes,
# so rebuilding after a code change takes seconds.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# Download the embedding model at build time, so the container starts fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY eval ./eval
COPY data/raw ./data/raw
COPY --from=ui /ui/dist ./ui/dist

CMD ["uvicorn", "app.api:api", "--host", "0.0.0.0", "--port", "8000"]
```

**`.dockerignore`**: keeps secrets and big folders out of the image.
```
# Keeps secrets and big folders out of the image (and makes builds faster).
.venv
.env
.git
data/qdrant
data/phoenix
eval/*/results
ui/node_modules
ui/dist
```

**Add the app to `docker-compose.yml`** (below the `phoenix` service):
```yaml
  app:                     # Doc 5: FastAPI + the React UI, built from the Dockerfile
    build: .
    profiles: ["app"]      # only started with --profile app, so Docs 1-4 keep running just Qdrant + Phoenix
    ports:
      - "8000:8000"        # UI at http://localhost:8000, API docs at /docs
    env_file: .env         # the same settings as outside Docker...
    environment:           # ...except the addresses, which differ inside a container:
      QDRANT_URL: http://qdrant:6333      # services in one Compose file find each other by name
      PHOENIX_URL: http://phoenix:6006
      LLM_BASE_URL: http://host.docker.internal:1234/v1   # LM Studio on your PC ("localhost" would be the container)
    depends_on:
      - qdrant
      - phoenix
```

Why override three addresses? Inside a container, `localhost` is the container itself, not your PC. Qdrant and Phoenix are other Compose services, so they're reached by service name. LM Studio runs on your PC, which Docker Desktop calls `host.docker.internal`.

Stop the uvicorn from step 3 (it uses port 8000), then build and start the whole stack:

```bash
docker compose --profile app up -d --build        # first build ~1-2 min, image ~1.1 GB
docker compose exec app python -m app.rag.ingest  # rebuild the Qdrant collection from inside the container
curl -s localhost:8000/api/health
# {"status":"ok","llm":"openai/gpt-oss-20b","llm_host":"host.docker.internal","chunks":304}
```

Open **http://localhost:8000**: the UI, served by the container. The header now says `on host.docker.internal`: the container is talking to LM Studio on your PC. Trace links still work, because Phoenix is published on your PC's port 6006.

`profiles: ["app"]` means a plain `docker compose up -d` (Docs 1-4) still starts only Qdrant and Phoenix. Stop the app with `docker compose --profile app down` (Qdrant and Phoenix keep running).

After changing code, add `--build` again: the image is a snapshot, like `ui/dist`. Python packages are cached in their own layer, so a rebuild after a code change is quick.

---

## 5. Run an eval in the container (10 min)

The container should answer exactly like your code outside it. Check it the usual way, with an eval:

```bash
docker compose exec app python -m eval.rag.evaluate
# MEAN  retrieval_hit=1.00  context_recall=0.91  faithfulness=0.86  answer_accuracy=0.86  latency_s=3.52
```

The scores should be close to your local RAG run (LLM judges are noisy: one question = ~10 points). The report is written inside the container; copy it out to open it:

```bash
docker compose cp app:/app/eval/rag/results ./eval/rag/results-docker
```

The other evals work the same way: `eval.single_agent.evaluate`, `eval.multi_agent.evaluate`, `eval.langgraph_agent.evaluate`. Add a row with "Where = Docker" to the README table.

---

## 6. How companies run the same thing

You built the simple version on purpose. Each row replaces one piece; the app code stays the same.

| You did | Companies usually use | Why |
|---|---|---|
| Docker Compose on your PC | The same image on **Kubernetes**, AWS ECS, Google Cloud Run or Azure Container Apps | auto-restart, scaling, no server to patch |
| UI served by FastAPI | UI on a CDN (e.g. S3 + CloudFront, Vercel), API separate | the page is cached worldwide; API and UI deploy independently |
| LM Studio on your GPU | A hosted model (Amazon Bedrock, Azure OpenAI, ...) or a GPU server running vLLM | no desktop app; scales with traffic. The OpenAI-style API means only `.env` changes |
| Qdrant container | Qdrant Cloud, OpenSearch, or Postgres + pgvector | managed backups, high availability |
| No login, local only | HTTPS + auth (SSO, API keys) in front of the API | security |
| Phoenix on your PC | Hosted tracing (Phoenix Cloud, Langfuse, LangSmith) | debug real user traffic, kept for months |
| One blocking HTTP call per answer | **Streaming** the answer token by token | users see text after 1 s, not 7 |
| Run evals by hand | **Evals in CI** as a gate before each release | no quality regressions |

---

## Checklist

- [x] I can explain why every system returns the same JSON shape, and what the adapters in `app/api.py` do.
- [x] I used the UI to see the difference between the four systems (steps, sources, time), and broke it on purpose once.
- [x] I can explain why the UI calls `/api/...` and how that works both with `npm run dev` and with the built UI.
- [x] I can explain what a container image is, why the Dockerfile has two stages, and why the container can't use `localhost` for LM Studio.
- [x] I ran an eval inside the container and got about the same scores as outside it.

**Done.** Back to the [README](../README.md) to fill in your results table.
