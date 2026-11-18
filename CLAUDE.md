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

## Stack (don't add to it without a reason)

LM Studio `openai/gpt-oss-20b` (local) / Bedrock `openai.gpt-oss-20b-1:0` (AWS), both through the OpenAI API · Qdrant in Docker · fastembed `BAAI/bge-small-en-v1.5` · PyMuPDF · LangChain `create_agent` · RAGAS 0.4 (`ragas.metrics.collections`) · FastAPI · Docker Compose on EC2.

## Commands (from repo root, always via `uv run`)

```bash
docker compose up -d                       # Qdrant
lms server start && lms load openai/gpt-oss-20b --context-length 32768 -y
uv run python -m app.ingest                # PDFs -> Qdrant
uv run python -m app.rag "question"        # also app.agent, app.multi_agent
uv run python -m eval.rag_eval             # also eval.agent_eval, eval.multi_agent_eval
uv add <pkg>                               # never pip
```

## Gotchas (tested 2026-11-18)

- **Windows + Git Bash:** console is cp1252, so scripts call `sys.stdout.reconfigure(encoding="utf-8")`.
- **Docker Desktop** must be running before `docker compose`.
- **RAGAS judge with LM Studio:** needs `mode=instructor.Mode.JSON_SCHEMA` (LM Studio rejects `json_object`) and `max_tokens=8192, reasoning_effort="low"` (gpt-oss reasons first and hits the default 1,024-token cap).
- `langchain-community` is being sunset: don't use it. qdrant-client does embeddings itself via `models.Document`.
- LLM judges are noisy: a correct answer sometimes scores 0. With ~10 questions, one question = 10 points.
- Bedrock path not yet tested end to end (no AWS access during setup); if the judge errors on Bedrock, try `instructor.Mode.TOOLS`.

## Current state (update at the end of each session)

- Fresh setup done: `pyproject.toml` (deps installed), `docker-compose.yml` (Qdrant), `.env.example`, question sets in `eval/`, 4 docs written. `app/` and `eval/` contain only `__init__.py` + the question files.
- Reference numbers from the setup test run (local): RAG hit 1.00 / recall 0.91 / faith 0.91 / acc 0.91; agent tool 0.75 / goal 0.75; multi-agent routing 0.88 / goal 0.88 (1.6× latency).
- **Next:** user starts Doc 1, step 2.
- **Git:** remote `rag-lab` → https://github.com/RVicky172/RAG-LAB.git (`main` tracks it). Commit only when asked; end messages with the Co-Authored-By line.
