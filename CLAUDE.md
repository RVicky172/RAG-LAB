# CLAUDE.md

Guidance for Claude in this repository. Read [README.md](README.md) first (plan, stack, results table).

## What this project is

A **one-day learning plan** in 4 docs (`docs/01..04-*.md`): RAG + eval → AI agent + eval → multi-agent + eval → deploy on AWS. The user is learning: the goal is understanding, kept simple. The docs contain the full, tested code; the user creates the files in `app/` and `eval/` by following them.

(Reset on 2026-11-18. The earlier 11-phase plan and `rag-lab/` code are in git history at commit `3867fca`.)

## How to work with the user

- **Keep it simple and practical.** Build first, then explain in a few plain sentences. No long theory blocks, no synthetic benchmarks, no extra libraries. The user said the old plan was too complicated and not how companies work.
- Follow the current doc step by step; say what the next step is. Don't jump ahead.
- When something fails, explain why in plain language.
- Every change ends with an eval run and a row in the README results table ("change one thing, measure").
- Read failing eval rows with the user: decide whether retrieval, the LLM or the judge was wrong.
- **Code style (user request, 2026-11-18):** every file starts with a docstring (what / why / how to run), every function has a Google-style docstring (`Args:` / `Returns:`), comments explain *why*, and the code does one step per line (no dense one-liners). Code in the docs must match the files in `app/` and `eval/`.

## Stack (don't add to it without a reason)

LM Studio `openai/gpt-oss-20b` (local) / Bedrock `openai.gpt-oss-20b-1:0` (AWS), both through the OpenAI API · Qdrant in Docker · fastembed `BAAI/bge-small-en-v1.5` · PyMuPDF · LangChain `create_agent` · RAGAS 0.4 (`ragas.metrics.collections`) · Arize Phoenix `version-21.2.0` in Docker for tracing (local, no account) · FastAPI · Docker Compose on EC2.

## Commands (from repo root, always via `uv run`)

```bash
docker compose up -d                       # Qdrant (:6333) + Phoenix (:6006)
lms server start && lms load openai/gpt-oss-20b --context-length 32768 -y
uv run python -m app.rag.ingest            # PDFs -> Qdrant
uv run python -m app.rag.query "question"  # also app.single_agent.agent, app.multi_agent.supervisor
uv run python -m eval.rag.evaluate          # also eval.single_agent.evaluate, eval.multi_agent.evaluate
uv run python -m eval.rebuild_reports       # rebuild all HTML reports from saved results
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
- Bedrock path not yet tested end to end (no AWS access during setup); if the judge errors on Bedrock, try `instructor.Mode.TOOLS`.

## Current state (update at the end of each session)

- Fresh setup done: `pyproject.toml` (deps installed), `docker-compose.yml` (Qdrant), `.env.example`, question sets in `eval/`, 4 docs written.
- Reference numbers from the setup test run (local): RAG hit 1.00 / recall 0.91 / faith 0.91 / acc 0.91; agent tool 0.75 / goal 0.75; multi-agent routing 0.88 / goal 0.88 (1.6× latency).
- **Layout (2026-11-18, user request):** `shared/` folders hold code used by more than one system/eval; one folder per system: `app/{shared,rag,single_agent,multi_agent}`, `eval/{shared,rag,single_agent,multi_agent}`. Each eval folder has its questions and writes `results/{results.json,report.html}`. The multi-agent eval reuses `eval/single_agent/` (main, diagnose, report). All Docs 1–3 code is written and tested. The user's `app/rag/ingest.py` uses chunk 800/80; Doc 1 keeps the tested 1000/150 (the doc-sync step swaps it).
- **Doc 4 (AWS) not updated on purpose (user asked to leave it):** its `app/api.py` still imports `from app import agent, multi_agent, rag` and runs `python -m app.ingest`. When we get to Doc 4, change these to `app.rag.query`, `app.single_agent.agent`, `app.multi_agent.supervisor`, `app.rag.ingest`.
- **Tracing (2026-11-18):** Phoenix replaced the manual trace/token code (agent token callback, judge httpx hook, trace_steps). `start_tracing(project)` in each entry point; projects `rag`, `single-agent`, `multi-agent`, `rag-eval`, `single-agent-eval`, `multi-agent-eval`. Evals wrap each question in one span and store `trace_id`; reports link to `/redirects/traces/<id>`. Only the RAG prompt split (`eval/rag/tokens.py`, tiktoken) stays custom. `TRACING=false` turns it off. Check report changes with headless Edge screenshots (`msedge --headless=new --screenshot=...`); the Chrome extension wasn't connected.
- **Next:** user starts Doc 1, step 2.
- **Git:** remote `rag-lab` → https://github.com/RVicky172/RAG-LAB.git (`main` tracks it). Commit only when asked; end messages with the Co-Authored-By line.
