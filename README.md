# RAG, Agents and Multi-Agent Systems: a one-day learning plan

Build three AI systems over the same 4 research papers, measure each one, then deploy them on AWS.

| # | Doc | You build | You measure | Time |
|---|-----|-----------|-------------|------|
| 1 | [RAG and evaluation](docs/01-rag-and-evaluation.md) | PDFs → Qdrant → answer with citations | retrieval hit, context recall, faithfulness, answer accuracy, tokens (HTML report) | 3 h |
| 2 | [AI agent and evaluation](docs/02-ai-agent-and-evaluation.md) | an agent with `search_papers` + `calculator` tools | tool use, goal accuracy, steps, latency | 2 h |
| 3 | [Multi-agent system and evaluation](docs/03-multi-agent-and-evaluation.md) | supervisor + researcher + analyst | routing, goal accuracy vs single agent, latency | 1.5 h |
| 4 | [Deploy on AWS](docs/04-deploy-on-aws.md) | FastAPI in Docker on EC2, LLM on Amazon Bedrock | same evals in the cloud; cost per question | 2 h |

## The stack (and why each piece)

| Need | Choice | Why this one |
|---|---|---|
| LLM (local) | **LM Studio** + `gpt-oss-20b` | free, runs on the RTX 5080 (12 GB), good at tool calling |
| LLM (AWS) | **Amazon Bedrock** + `gpt-oss-20b` | the same model, the same OpenAI-style API: only `.env` changes |
| Vector database | **Qdrant** (Docker) | widely used in industry, web dashboard, embeds text for you |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed | small, free, runs on CPU inside qdrant-client |
| PDF → text | **PyMuPDF** | fast, accurate, no models |
| Agents | **LangChain `create_agent`** | the standard agent loop in a single function |
| Evaluation | **RAGAS** + plain Python checks | the standard RAG/agent metrics library |
| Tracing | **Arize Phoenix** (Docker) | see every LLM / tool call with tokens and timing; runs locally, no account |
| API + deploy | **FastAPI**, **Docker Compose**, **EC2** | simplest deployment you can fully see |

## Setup (once)

Needs: [uv](https://docs.astral.sh/uv/), Docker Desktop, LM Studio (with the `lms` CLI).

```bash
uv sync                                     # Python packages
cp .env.example .env                        # settings
docker compose up -d                        # Qdrant -> :6333/dashboard, Phoenix -> http://localhost:6006
lms server start
lms load openai/gpt-oss-20b --context-length 32768 -y
```

Then start [Doc 1](docs/01-rag-and-evaluation.md).

## Project layout

```
data/raw/            the knowledge base: 4 arXiv PDFs (Attention, RAG, Lost in the Middle, RAGAS)
app/shared/          code used by more than one system: config, llm, tools, trace
app/rag/             Doc 1: ingest.py (PDFs -> Qdrant), query.py (retrieve -> augment -> generate)
app/single_agent/    Doc 2: agent.py (one agent with tools)
app/multi_agent/     Doc 3: specialists.py (researcher + analyst), supervisor.py
eval/shared/         code used by every eval: judge, questions, tokens, status, HTML report blocks
eval/rag/            Doc 1 eval: questions.jsonl, evaluate.py, results/ (git-ignored)
eval/single_agent/   Doc 2 eval: questions.jsonl, evaluate.py, results/
eval/multi_agent/    Doc 3 eval: evaluate.py (reuses the single-agent eval), results/
docs/                the 4 learning docs
docker-compose.yml   Qdrant + Phoenix (+ the app, added in Doc 4)
.env                 settings and secrets (git-ignored; template in .env.example)
```

## Results

Fill one row per eval run. Change one thing at a time.

| Date | System | Where | Change | Metrics | Latency (s) |
|------|--------|-------|--------|---------|-------------|
| | RAG | local | baseline (k=5, chunk 1000/150) | hit= recall= faith= acc= | |
| | Agent | local | baseline | tool= goal= steps= | |
| | Multi-agent | local | baseline | routing= goal= steps= | |
| | RAG | AWS | Bedrock | hit= recall= faith= acc= | |
| | Agent | AWS | Bedrock | tool= goal= steps= | |
| | Multi-agent | AWS | Bedrock | routing= goal= steps= | |

## Progress

- [ ] Doc 1: RAG and evaluation
- [ ] Doc 2: AI agent and evaluation
- [ ] Doc 3: Multi-agent system and evaluation
- [ ] Doc 4: Deploy on AWS
