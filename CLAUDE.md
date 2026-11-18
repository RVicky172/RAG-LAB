# CLAUDE.md

Guidance for Claude in this repository. Read [README.md](README.md) first (plan, stack, results table).

## What this project is

A **learning plan** in 5 docs (`docs/01..05-*.md`): RAG + eval → AI agent + eval → multi-agent + eval → LangGraph agent + eval → API + React UI + Docker (local only; no AWS, user decision 2026-11-18). The user is learning: the goal is understanding, kept simple. The docs contain the full, tested code; the user creates the files in `app/`, `eval/` and `ui/` by following them.

(Reset on 2026-11-18. The earlier 11-phase plan and `rag-lab/` code are in git history at commit `3867fca`.)

## How to work with the user

- **Keep it simple and practical.** Build first, then explain in a few plain sentences. No long theory blocks, no synthetic benchmarks, no extra libraries. The user said the old plan was too complicated and not how companies work.
- Follow the current doc step by step; say what the next step is. Don't jump ahead.
- When something fails, explain why in plain language.
- Every change ends with an eval run and a row in the README results table ("change one thing, measure").
- Read failing eval rows with the user: decide whether retrieval, the LLM or the judge was wrong.
- **Code style (user request, 2026-11-18):** every file starts with a docstring (what / why / how to run), every function has a Google-style docstring (`Args:` / `Returns:`), comments explain *why*, and the code does one step per line (no dense one-liners). Code in the docs must match the files in `app/` and `eval/`.

## Stack (don't add to it without a reason)

LM Studio `openai/gpt-oss-20b` (local) through the OpenAI API · Qdrant in Docker · fastembed `BAAI/bge-small-en-v1.5` · PyMuPDF · LangChain `create_agent` · RAGAS 0.4 (`ragas.metrics.collections`) · Arize Phoenix `version-21.2.0` in Docker for tracing (local, no account) · FastAPI · React 19 + Vite 8 (plain JS + CSS, no UI libraries) · Docker Compose.

## Commands (from repo root, always via `uv run`)

```bash
docker compose up -d                       # Qdrant (:6333) + Phoenix (:6006)
lms server start && lms load openai/gpt-oss-20b --context-length 32768 -y
uv run python -m app.rag.ingest            # PDFs -> Qdrant
uv run python -m app.rag.query "question"  # also app.single_agent.agent, app.multi_agent.supervisor, app.langgraph_agent.graph
uv run python -m eval.rag.evaluate          # also eval.single_agent.evaluate, eval.multi_agent.evaluate, eval.langgraph_agent.evaluate
uv run python -m eval.rebuild_reports       # rebuild all HTML reports from saved results
uv run uvicorn app.api:api --port 8000     # API (+ built UI at /), docs at /docs
cd ui && npm run dev                       # UI dev server :5173 (proxies /api to :8000); npm run build -> ui/dist
docker compose --profile app up -d --build # app container (API + UI) on :8000
uv add <pkg>                               # never pip
```

## Gotchas (tested 2026-11-18)

- **Windows + Git Bash:** console is cp1252, so scripts call `sys.stdout.reconfigure(encoding="utf-8")`.
- **Docker Desktop** must be running before `docker compose`.
- **RAGAS judge with LM Studio:** needs `mode=instructor.Mode.JSON_SCHEMA` (LM Studio rejects `json_object`) and `max_tokens=8192, reasoning_effort="low"` (gpt-oss reasons first and hits the default 1,024-token cap).
- `langchain-community` is being sunset: don't use it. qdrant-client does embeddings itself via `models.Document`.
- **Phoenix 21 turns login on by default** inside Docker: compose sets `PHOENIX_ENABLE_AUTH=false`, binds `127.0.0.1:6006`, and turns off telemetry and the agent assistant. Don't remove the 127.0.0.1 binding (no login).
- **Tracing double-count:** the OpenAI instrumentor also catches LangChain's own (sync) OpenAI calls. `app/shared/tracing.py` instruments LangChain always and OpenAI only for the async client (the RAGAS judge), by restoring `openai.OpenAI.request`. If the app ever uses async LangChain calls, revisit this.
- LLM judges are noisy: a correct answer sometimes scores 0. With ~10 questions, one question = 10 points.
- **App container:** inside Docker `localhost` is the container, so compose overrides `QDRANT_URL`, `PHOENIX_URL` and `LLM_BASE_URL=http://host.docker.internal:1234/v1` (LM Studio on the PC). The `app` service has `profiles: ["app"]` so plain `docker compose up -d` still starts only Qdrant + Phoenix.
- Doc 5 code blocks are generated from the real files (app/api.py, ui/*, Dockerfile, .dockerignore, compose app service): after changing one, re-sync the doc.

## Current state (update at the end of each session)

- Fresh setup done: `pyproject.toml` (deps installed), `docker-compose.yml` (Qdrant), `.env.example`, question sets in `eval/`, 4 docs written.
- Reference numbers from the setup test run (local): RAG hit 1.00 / recall 0.91 / faith 0.91 / acc 0.91; agent tool 0.75 / goal 0.75; multi-agent routing 0.88 / goal 0.88 (1.6× latency).
- **Layout (2026-11-18, user request):** `shared/` folders hold code used by more than one system/eval; one folder per system: `app/{shared,rag,single_agent,multi_agent}`, `eval/{shared,rag,single_agent,multi_agent}`. Each eval folder has its questions and writes `results/{results.json,report.html}`. The multi-agent eval reuses `eval/single_agent/` (main, diagnose, report). All Docs 1–3 code is written and tested. The user's `app/rag/ingest.py` uses chunk 800/80; Doc 1 keeps the tested 1000/150 (the doc-sync step swaps it).
- **Doc 5 = API, UI and Docker (2026-11-18; user said don't deploy on AWS, so all AWS/Bedrock/EC2 content was removed; file renamed `docs/05-api-ui-and-docker.md`).** `app/api.py`: `GET /api/health`, `POST /api/ask/{system}` (rag | agent | multi-agent | langgraph) → `{answer, sources, steps, seconds, trace_id}` via one adapter per system; serves `ui/dist` at `/` if built. `ui/`: React + Vite, files `vite.config.js` (proxy /api), `src/{main,App,AnswerCard}.jsx`, `src/{api,systems}.js`, `src/styles.css`; modes per system + "Compare all"; no chat memory (systems are single-question). Two-stage `Dockerfile` (node:24-slim builds UI, python:3.12-slim + uv 0.13.6), image ~1.1 GB. Tested: all four systems via API, UI in headless Edge (CDP script), container ingest + RAG eval (hit 1.00 / recall 0.91 / faith 0.86 / acc 0.86).
- **Doc 4 LangGraph agent (2026-11-18):** `app/langgraph_agent/{nodes,graph}.py`, self-correcting RAG (retrieve → grade → rewrite → generate, MAX_SEARCHES=2). Eval reuses `eval/rag/evaluate.main(answer_fn=...)`. Tested: recall 0.79 / faith 0.95 / acc 0.86 vs RAG 0.73 / 0.87 / 0.86; 2.4 LLM calls and ~2,600 tokens per question vs 1.0 / ~1,200; 7.5 s vs 4.8 s. First version was worse (strict grader, keyword-style rewrites, dropped first chunks); Doc 4 §3.4 tells that story.
- **Tracing (2026-11-18):** Phoenix replaced the manual trace/token code (agent token callback, judge httpx hook, trace_steps). `start_tracing(project)` in each entry point; projects `rag`, `single-agent`, `multi-agent`, `langgraph`, `rag-eval`, `single-agent-eval`, `multi-agent-eval`, `langgraph-eval`. Evals wrap each question in one span and store `trace_id`; reports link to `/redirects/traces/<id>`. Only the RAG prompt split (`eval/rag/tokens.py`, tiktoken) stays custom. `TRACING=false` turns it off. Check report changes with headless Edge screenshots (`msedge --headless=new --screenshot=...`); the Chrome extension wasn't connected.
- **Progress (2026-11-18):** user verified Docs 1-5 and all checklist items are ticked (docs + README Progress). README results table filled from eval/*/results/results.json (all runs used chunk 800/80; Docker RAG row from the in-container eval).
- **Git:** remote `rag-lab` → https://github.com/RVicky172/Research-AI-Agent.git (renamed from RAG-LAB) (`main` tracks it). Commit only when asked; end messages with the Co-Authored-By line.
