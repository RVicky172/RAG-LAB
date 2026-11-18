# RAG, Agents and Multi-Agent Systems: a one-day learning plan

Build four AI systems over the same 4 research papers, measure each one, then put them behind one API with a web UI, in Docker.

| # | Doc | You build | You measure | Time |
|---|-----|-----------|-------------|------|
| 1 | [RAG and evaluation](docs/01-rag-and-evaluation.md) | PDFs → Qdrant → answer with citations | retrieval hit, context recall, faithfulness, answer accuracy, tokens (HTML report) | 3 h |
| 2 | [AI agent and evaluation](docs/02-ai-agent-and-evaluation.md) | an agent with `search_papers` + `calculator` tools | tool use, goal accuracy, steps, latency | 2 h |
| 3 | [Multi-agent system and evaluation](docs/03-multi-agent-and-evaluation.md) | supervisor + researcher + analyst | routing, goal accuracy vs single agent, latency | 1.5 h |
| 4 | [LangGraph agent and evaluation](docs/04-langgraph-agent-and-evaluation.md) | a self-correcting RAG graph: retrieve → grade → rewrite → generate | same RAG scores vs Doc 1, extra LLM calls, latency | 1.5 h |
| 5 | [API, UI and Docker](docs/05-api-ui-and-docker.md) | one FastAPI for all four systems, a React UI to talk to them, both in one container | same eval inside the container | 2 h |

## The stack (and why each piece)

| Need | Choice | Why this one |
|---|---|---|
| LLM (local) | **LM Studio** + `gpt-oss-20b` | free, runs on the RTX 5080 (12 GB), good at tool calling |
| Vector database | **Qdrant** (Docker) | widely used in industry, web dashboard, embeds text for you |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed | small, free, runs on CPU inside qdrant-client |
| PDF → text | **PyMuPDF** | fast, accurate, no models |
| Agents | **LangChain `create_agent`**, **LangGraph** | the standard agent loop in one function; LangGraph when you design the flow yourself |
| Evaluation | **RAGAS** + plain Python checks | the standard RAG/agent metrics library |
| Tracing | **Arize Phoenix** (Docker) | see every LLM / tool call with tokens and timing; runs locally, no account |
| API + UI | **FastAPI**, **React** + **Vite** | the usual company pair: a typed Python API and a component-based web page |
| Packaging | **Docker** + **Docker Compose** | one image with everything; runs the same on any machine |

## Setup (once)

Needs: [uv](https://docs.astral.sh/uv/), Docker Desktop, LM Studio (with the `lms` CLI). Doc 5 also needs [Node.js](https://nodejs.org) LTS for the UI.

```bash
uv sync                                     # Python packages
cp .env.example .env                        # settings
docker compose up -d                        # Qdrant -> :6333/dashboard, Phoenix -> http://localhost:6006
lms server start
lms load openai/gpt-oss-20b --context-length 32768 -y
```

Then start [Doc 1](docs/01-rag-and-evaluation.md).

## Run the app (UI + API)

Built in [Doc 5](docs/05-api-ui-and-docker.md). The UI is a web page that talks to the API (`app/api.py`), and the API uses the same Qdrant and LM Studio as everything else. So **three things must be running first**:

```bash
docker compose up -d                                   # 1. Qdrant + Phoenix (Docker Desktop must be running)
lms server start                                       # 2. LM Studio with the model loaded
lms load openai/gpt-oss-20b --context-length 32768 -y
uv run python -m app.rag.ingest                        # 3. chunks in Qdrant (once; they survive restarts)
cd ui && npm install && cd ..                          #    UI packages (once)
```

Then start the app in **one** of three ways:

| Way | Commands | Open | Use it when |
|---|---|---|---|
| **A. One server** | `cd ui && npm run build && cd ..`<br>`uv run uvicorn app.api:api --port 8000` | http://localhost:8000 | you just want to use the app |
| **B. Two servers** | terminal 1: `uv run uvicorn app.api:api --port 8000`<br>terminal 2: `cd ui && npm run dev` | http://localhost:5173 | you're changing UI code (the page reloads on save) |
| **C. Docker** | `docker compose --profile app up -d --build` | http://localhost:8000 | you want to run it the way it would be shipped |

**Stop it:** Ctrl+C in the uvicorn / npm terminals (A and B), or `docker compose --profile app down` (C). A and C both use port 8000, so run only one of them at a time.

| URL | What |
|---|---|
| http://localhost:8000 (or :5173) | the UI: pick a system or "Compare all", ask, see answer + steps + sources |
| http://localhost:8000/docs | FastAPI's test page for the API |
| http://localhost:6006 | Phoenix: every trace (the UI's "Open trace" links go here) |
| http://localhost:6333/dashboard | Qdrant: the stored chunks |

**Something wrong?** The UI says what in its header or on the answer card: *API not reachable* → uvicorn isn't running; *Qdrant is empty* → run the ingest; *Connection error* on a card → LM Studio isn't running or has no model loaded; `{"detail":"Not Found"}` at :8000 → build the UI, then restart uvicorn. More in [Doc 5, §3.5](docs/05-api-ui-and-docker.md#35-when-something-doesnt-work).

## Project layout

```
data/raw/            the knowledge base: 4 arXiv PDFs (Attention, RAG, Lost in the Middle, RAGAS)
app/shared/          code used by more than one system: config, llm, tools, trace
app/rag/             Doc 1: ingest.py (PDFs -> Qdrant), query.py (retrieve -> augment -> generate)
app/single_agent/    Doc 2: agent.py (one agent with tools)
app/multi_agent/     Doc 3: specialists.py (researcher + analyst), supervisor.py
app/langgraph_agent/ Doc 4: nodes.py (state + steps), graph.py (the LangGraph graph)
app/api.py           Doc 5: FastAPI, one endpoint for all four systems (+ serves the built UI)
ui/                  Doc 5: React + Vite UI (npm run dev -> :5173, npm run build -> ui/dist)
eval/shared/         code used by every eval: judge, questions, tokens, status, HTML report blocks
eval/rag/            Doc 1 eval: questions.jsonl, evaluate.py, results/ (git-ignored)
eval/single_agent/   Doc 2 eval: questions.jsonl, evaluate.py, results/
eval/multi_agent/    Doc 3 eval: evaluate.py (reuses the single-agent eval), results/
eval/langgraph_agent/ Doc 4 eval: evaluate.py (reuses the RAG eval), results/
docs/                the 5 learning docs
Dockerfile           Doc 5: builds the UI, then the Python app image
docker-compose.yml   Qdrant + Phoenix (+ the app, profile "app", Doc 5)
.env                 settings and secrets (git-ignored; template in .env.example)
```

## Results

Fill one row per eval run. Change one thing at a time.

| Date | System | Where | Change | Metrics | Latency (s) |
|------|--------|-------|--------|---------|-------------|
| 2026-11-18 | RAG | local | baseline (k=5, chunk 800/80) | hit=1.00 recall=0.73 faith=0.87 acc=0.86 | 4.8 |
| 2026-11-18 | Agent | local | baseline | tool=0.88 goal=0.75 steps=1.00 | 1.8 |
| 2026-11-18 | Multi-agent | local | baseline | routing=0.88 goal=0.75 steps=1.00 | 2.9 |
| 2026-11-18 | LangGraph agent | local | baseline (MAX_SEARCHES=2) | hit=1.00 recall=0.79 faith=0.95 acc=0.86 searches=1.18 | 7.5 |
| 2026-11-18 | RAG | Docker | same code in the container | hit=1.00 recall=0.91 faith=0.86 acc=0.86 | 3.5 |

What the numbers say:
- **LangGraph vs RAG:** the grade-and-retry loop raised recall (0.73 → 0.79) and faithfulness (0.87 → 0.95) at the same accuracy, for ~1.6× the latency (it searched twice on 2 of 11 questions, plus a grading call on every one).
- **Multi-agent vs agent:** same routing and goal scores, ~1.6× slower: every delegation runs a whole specialist agent.
- **Docker vs local RAG:** same code, same chunks, yet recall 0.91 vs 0.73. With 11 questions one question moves a score by ~9 points, so this is LLM and judge noise, not a real difference. Run each eval 2-3 times before trusting a gap this small.

## Progress

- [x] Doc 1: RAG and evaluation
- [x] Doc 2: AI agent and evaluation
- [x] Doc 3: Multi-agent system and evaluation
- [x] Doc 4: LangGraph agent and evaluation
- [x] Doc 5: API, UI and Docker
