# RAG Mastery: A Learning Plan

A step-by-step plan to go from "what is RAG?" to designing, building, evaluating and running production Retrieval-Augmented Generation systems, in the order a working AI engineer learns and uses these skills.

Each phase has its own folder under [`docs/`](docs/) with a `README.md` that covers **what to learn, why it matters, hands-on labs, a milestone deliverable, an exit checklist and interview questions**. Work through them in order. Don't skip a milestone: each one is the code base you build the next phase on.

> **Companion document:** [RAG_Production_Mastery_Guide.docx](docs/RAG_Production_Mastery_Guide.docx) is your reference and interview book (AWS-focused). Every phase README links to the matching sections of it. Phases 9 and 10 follow its Part 3 (projects) and Part 4 (interview prep).

> **Model guide:** [MODELS.md](docs/MODELS.md) says which local model plays which role, why it was chosen, what problem it solves, how to set it up in LM Studio, and what replaces it on AWS Bedrock. **Its §0 "Final model checklist"** lists all 7 LM Studio models (✅ all installed, 58.45 GB) and the 5 Python models that download on first use.

## Strategy: local first, then AWS

```
 Phases 00–08: LOCAL (free, private, unlimited iterations)         Phase 09: AWS
 ┌───────────────────────────────────────────────────────┐        ┌──────────────────────────────┐
 │ LM Studio (OpenAI-compatible API, localhost:1234)     │        │ Amazon Bedrock               │
 │   Generator  Qwen3-30B-A3B-Instruct-2507              │  same  │   Claude / Nova / open models │
 │   Vision     Qwen3.6-35B-A3B (+mmproj)                │  code, │   Titan V2 / Cohere Embed     │
 │   Helper     Qwen3-4B-Instruct-2507                   │ change │   Bedrock Rerank              │
 │   Judge      gpt-oss-20b                              │  .env  │   Knowledge Bases, Guardrails │
 │   Embeddings nomic-embed / Qwen3-Embedding            │ ─────► │ OpenSearch / S3 Vectors       │
 │ Python GPU: bge-m3, bge-reranker-v2-m3, ColQwen       │        │ Lambda, Fargate, Step Fns     │
 │ Docker: Qdrant / OpenSearch / pgvector, Phoenix       │        │ CloudWatch, X-Ray             │
 └───────────────────────────────────────────────────────┘        └──────────────────────────────┘
```

All code calls models through two thin wrappers (`rag/llm.py`, `rag/embed.py`, built in Phase 01) that switch provider via `.env`. You understand everything locally, then migrate by configuration, re-embed, and re-run the same golden set to compare.

---

## Quick start (local environment)

Full explanation and troubleshooting: [Phase 00, Step 1](docs/00-prerequisites/README.md#step-1-python-environment).

### Setup status (verified on this machine)

| Item | Status | Details |
|------|--------|---------|
| GPU / RAM | ✅ | RTX 5080 (16 GB VRAM), 127 GB RAM |
| Python env (`rag-lab/.venv`) | ✅ | Python 3.12.13 via uv; `torch 2.11.0+cu128`, CUDA available |
| Base packages | ✅ | openai, sentence-transformers, transformers, numpy, pandas, pydantic, python-dotenv, tqdm; dev: pytest, ipykernel |
| LM Studio chat models | ✅ 4/4 | Qwen3-30B-A3B-2507 (generator), Qwen3-4B-2507 (helper), Qwen3.6-35B-A3B (vision), gpt-oss-20b (judge) |
| LM Studio embedding models | ✅ 3/3 | nomic-embed-text-v1.5, Qwen3-Embedding-0.6B, Qwen3-Embedding-4B |
| `rag-lab/.env` | ✅ | LM Studio URL + exact model names (git-ignored) |
| Python models (rerankers, bge-m3, ColQwen) | ⏳ | Download automatically the first time Phases 01/03/05/07 use them |
| Phase-specific libraries | ⏳ | Added with `uv add` when each phase starts (pypdf, faiss, rank-bm25, docling, fastapi…) |
| LM Studio server | ▶️ Start each session | `lms server start` (or Developer tab → Start Server), then load models |

**Everything needed to start Phase 00/01 is in place.** Each session: start the LM Studio server, load the generator with a 32K context, and run labs with `uv run`.

### Commands

```bash
# 1. Python env: uv + Python 3.12 + CUDA PyTorch (✅ already done here; on a new machine just run `uv sync`)
cd /d/CodeBase/AI/RAG/rag-lab
uv sync                                    # creates .venv on 3.12, installs torch 2.11.0+cu128 (~3 GB first time)
uv add openai numpy pandas python-dotenv pydantic tqdm sentence-transformers transformers
uv add --dev pytest ipykernel
uv run python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
# expected: 3.12.x 2.11.0+cu128 True

# 2. Local models: start LM Studio's server and load the generator with a real context length
lms server start
lms load qwen/qwen3-30b-a3b-2507 --context-length 32768
curl http://localhost:1234/v1/models
```

Key decisions, already applied in `rag-lab/`:

| File | Setting | Why |
|------|---------|-----|
| `.python-version` | `3.12` | uv would otherwise pick the newest installed Python (3.15), which PyTorch doesn't support yet |
| `pyproject.toml` | `requires-python = ">=3.12,<3.13"` | Keeps the lock file on 3.12 |
| `pyproject.toml` | `[[tool.uv.index]] pytorch-cu128` + `[tool.uv.sources] torch` | CUDA 12.8 build for the RTX 5080; the default PyPI torch is CPU-only |

> Use `uv run <command>` instead of activating the venv. If you do activate it in Git Bash, use `source .venv/Scripts/activate` (forward slashes, with `source`).

---

## The roadmap at a glance

| # | Phase | Goal in one line | Time (part-time) | Milestone you ship |
|---|-------|------------------|------------------|--------------------|
| 00 | [Prerequisites & Local AI Setup](docs/00-prerequisites/README.md) | Be fluent with the tools before the theory | 1 week | LM Studio server + models running, first local LLM call, token & latency intuition |
| 01 | [Foundations & Naive RAG](docs/01-foundations-naive-rag/README.md) | Understand why RAG exists; build it from raw parts | 1–2 weeks | `naive_rag.py` with no frameworks + a 20-question golden set |
| 02 | [Ingestion & Chunking](docs/02-ingestion-chunking/README.md) | Fix "garbage in, garbage out" | 2 weeks | Layout-aware parser + 4 chunkers compared on the golden set |
| 03 | [Vector Search & Indexing](docs/03-vector-search-indexing/README.md) | Understand how search scales; add hybrid search | 2 weeks | HNSW/IVF benchmark + hybrid (BM25 + dense + RRF) retriever |
| 04 | [Query Transformation](docs/04-query-transformation/README.md) | Bridge the gap between user wording and document wording | 1 week | Query rewriter, multi-query, HyDE, router behind one interface |
| 05 | [Post-Retrieval & Reranking](docs/05-post-retrieval-reranking/README.md) | Send only high-signal context to the LLM | 1–2 weeks | Two-stage retrieval with cross-encoder + parent-child expansion |
| 06 | [Evaluation & Observability](docs/06-evaluation-observability/README.md) | Replace "vibes" with numbers | 2 weeks | Eval harness (retrieval + generation metrics) as a CI gate, tracing |
| 07 | [Advanced Paradigms](docs/07-advanced-paradigms/README.md) | Solve what top-k similarity cannot | 3 weeks | Agentic RAG + one of GraphRAG / multimodal / CRAG, measured |
| 08 | [Production Engineering](docs/08-production-engineering/README.md) | Make it secure, fresh, fast, affordable | 2 weeks | Idempotent ingestion, ACL filters, injection defense, caching, API |
| 09 | [Capstone Projects on AWS](docs/09-capstone-projects/README.md) | Build two portfolio systems end to end | 3–4 weeks | PolicyPal (managed) + DocuMind (custom hybrid multimodal) |
| 10 | [Interview Prep & Career](docs/10-interview-prep/README.md) | Explain and defend every decision | Ongoing | Ablation table, system-design script, 30 answers out loud |

Total: roughly **4–5 months part-time** (8–10 hours/week), or about 8 weeks full-time.

---

## What I changed from your original plan (and why)

Your 7-phase plan was strong. These are the enhancements and the reasoning behind each:

1. **Added Phase 00 (Prerequisites).** Tokens, context windows, API cost and a first LLM call. Without these the later "chunk size vs context window" trade-offs are abstract.
2. **Evaluation starts in Phase 01, not Phase 07.** Real AI engineers build a small golden set *before* they optimize. Otherwise you can't tell whether semantic chunking or a reranker actually helped. Phase 01 creates 20 questions; Phase 06 turns it into a full harness. Every phase from 02 onward ends with "measure it against the golden set".
3. **Evaluation (06) moved before Advanced Paradigms (07).** GraphRAG and agentic RAG are expensive and slow. You should only adopt them when your evaluation shows a failure class (multi-hop, global questions) that simpler methods can't fix.
4. **Split production work into its own phase (08).** Your Phase 7 mixed evaluation with security and operations. Production engineering (idempotent ingestion, incremental updates, deletions, ACLs, prompt injection, caching, cost and latency) is a full discipline. It's also what separates a tutorial from a hireable portfolio.
5. **Added missing modern techniques:**
   - *Contextual Retrieval* (Phase 02): an LLM-written context sentence per chunk. Anthropic's published benchmark cut retrieval failures by about 49%, and by about 67% with a reranker.
   - *Matryoshka embeddings, embedding fine-tuning and ColBERT/late interaction* (Phases 03 and 05).
   - *Long-context vs RAG* decision-making (Phase 01): modern models have 1M-token windows, so knowing when *not* to use RAG is a core skill.
   - *MMR (Maximal Marginal Relevance)* for diversity (Phase 05).
   - *Citations and grounded refusal* ("I don't know") as first-class features (Phase 01 onward).
6. **Added Phase 09 (Capstones) and Phase 10 (Interview prep)** from your Word guide, so the plan ends with two deployable AWS projects and a polished way to talk about them.
7. **Local first, cloud second.** Phases 00–08 run entirely on your machine with LM Studio and open-weight models (no API key, no bill). Phase 09 moves the same code to Bedrock by configuration, and you compare the results on the same golden set.

---

## How to work through each phase

Every phase README follows the same structure. Use it as a loop:

```
1. Read "Why this phase matters"  ──► know the problem before the solution
2. Study the concepts in order     ──► take notes in your own words (notes.md)
3. Do the labs                     ──► type the code; don't copy-paste it
4. Ship the milestone              ──► commit it to your repo
5. Measure against the golden set  ──► record numbers in results.md
6. Pass the exit checklist         ──► if you can't explain it, revisit it
7. Answer the interview questions out loud (under 1 minute each)
```

**The rule real engineers follow: change one thing, measure, keep it or revert it.**

---

## Recommended repository layout

Build one growing code base rather than disconnected notebooks:

```
RAG/
├── README.md                     ← this roadmap
├── docs/                         ← the learning plan: one folder per phase (notes + labs)
│   ├── MODELS.md                 ← model guide
│   ├── RAG_Production_Mastery_Guide.docx
│   ├── 00-prerequisites/README.md
│   ├── 01-foundations-naive-rag/README.md
│   ├── ...
│   └── 10-interview-prep/README.md
└── rag-lab/                      ← YOUR code, grows phase by phase (full map: rag-lab/README.md)
    ├── pyproject.toml            ← dependencies + CUDA torch index (uv)
    ├── uv.lock                   ← exact versions (commit it)
    ├── .python-version           ← 3.12 (commit it)
    ├── .venv/                    ← created by `uv sync` (git-ignored)
    ├── .env                      ← provider + model settings, later AWS profile (git-ignored!)
    ├── src/rag/                  ← THE LIBRARY (installed package `rag`)
    │   ├── llm.py  embed.py  vector_store.py   ← Phase 01
    │   ├── parse.py  chunk.py  enrich.py       ← Phase 02
    │   ├── index.py  search.py   ← Phase 03
    │   ├── query.py              ← Phase 04
    │   ├── rerank.py context.py  ← Phase 05
    │   ├── pipeline.py           ← the end-to-end RAG chain
    │   └── agent.py              ← Phase 07
    ├── labs/                     ← one-off learning scripts, one folder per phase
    │   ├── phase00_prerequisites/
    │   ├── phase01_naive_rag/    ← lab1a_embeddings, lab1c_naive_rag, lab1d_vector_stores, visualize
    │   └── phase02_ingestion/    ← lab2a_parsing
    ├── eval/
    │   ├── golden.jsonl          ← starts in Phase 01, grows forever
    │   ├── eval_core.py          ← shared scoring (hit@5, MRR)
    │   ├── results.md            ← summary: best config per phase
    │   ├── phase01_naive/        ← scripts, results.md, failures.md
    │   └── phase02_chunking/     ← scripts, results.md (ablation table), runs.jsonl
    ├── data/                     ← git-ignored
    │   ├── raw/                  ← source PDFs
    │   ├── processed/            ← parsed elements, chunks, cached vectors
    │   └── indexes/phaseNN/      ← vector indexes per phase
    ├── api/                      ← FastAPI service (Phase 08)
    └── tests/
```

## Default tech stack (free or cheap for learning)

Details and reasoning for every model: [MODELS.md](docs/MODELS.md).

| Layer | Learning default (local) | Production / AWS equivalent (Phase 09) |
|-------|--------------------------|----------------------------------------|
| Language | Python 3.12 (installed and pinned by **uv**) | same |
| Packages / environments | **uv** (`uv add`, `uv run`, `uv.lock`) | same (`uv sync` in Docker/CI) |
| Web server (Phase 08) | **uvicorn** running FastAPI | uvicorn in a container on ECS Fargate |
| Model server | **LM Studio** (OpenAI-compatible API via the `openai` package) | Amazon Bedrock (Converse API via `boto3`) |
| LLM (generation) | **Qwen3-30B-A3B-Instruct-2507** (installed) | Claude on Bedrock (or open-weight models on Bedrock) |
| LLM (helper calls) | **Qwen3-4B-Instruct-2507** | Claude Haiku tier / Amazon Nova Micro |
| Vision LLM | **Qwen3.6-35B-A3B** + mmproj (installed) | Claude vision / Amazon Nova |
| LLM judge | **gpt-oss-20b** (different family from the generator) | Bedrock Evaluations or a different-family model |
| Embeddings | **nomic-embed-text-v1.5** → **Qwen3-Embedding-0.6B** (LM Studio); **bge-m3** (Python) | Titan Text Embeddings V2, Cohere Embed |
| Vector store | NumPy, then FAISS and Chroma, then Qdrant (Docker) | OpenSearch, pgvector, S3 Vectors |
| Keyword search | `rank_bm25` | OpenSearch BM25 |
| Reranker | **BAAI/bge-reranker-v2-m3** (Python, GPU) | Bedrock Rerank (Amazon Rerank 1.0, Cohere Rerank 3.5) |
| Parsing | PyMuPDF, then Docling / Unstructured | Textract, Bedrock KB FM parser |
| Evaluation | your own scripts, then RAGAS / DeepEval | Bedrock Evaluations |
| Tracing | Arize Phoenix or Langfuse (local) | CloudWatch, X-Ray, Langfuse |
| API | FastAPI | ECS Fargate / Lambda |

> **Frameworks rule:** no LangChain or LlamaIndex until you have built the same thing by hand (end of Phase 05). After that, learn one of them so you can read other people's code. You should understand what every abstraction does underneath.

## Pick your practice corpus now

Choose **one** document set and keep it for the whole journey, so your metrics stay comparable:

- **Recommended:** 20–50 PDFs from one domain you care about, such as company policies, a product's technical manuals, annual reports (they have tables and charts) or research papers from arXiv.
- Must include: multi-column pages, at least 5 tables, some figures, product codes or IDs, and documents that change over time (two versions of the same policy).
- Avoid toy text files. Messy real data is the whole point of Phase 02.

---

## Progress tracker

Tick these off as you go:

- [x] 00 Prerequisites: LM Studio server, models loaded, first local LLM call, tokens and latency
- [x] 01 Naive RAG from scratch + 20-question golden set (21 q, golden v2; baseline hit@5 0.78 / MRR 0.72)
- [ ] 02 Parsing + 4 chunking strategies compared (**in progress**, details in [Phase 02 results](rag-lab/eval/phase02_chunking/results.md))
  - [x] Step 1: look at the data, defects log ([defects.md](rag-lab/labs/phase02_ingestion/lab2a_parsing/defects.md))
  - [x] Lab 2a: parser shoot-out, pypdf vs PyMuPDF vs Docling ([shootout.md](rag-lab/labs/phase02_ingestion/lab2a_parsing/shootout.md)) → Docling
  - [x] `rag/parse.py`: typed elements, page + section path, cleaning, figure crops
  - [x] `rag/chunk.py`: recursive, structure-aware (table row-split + header), semantic; `rag/enrich.py`: contextual retrieval
  - [x] Step 4: chunk metadata schema with deterministic ids
  - [x] Step 6: ablation on NumPy / FAISS / Chroma → Docling + structure-aware 0.83 / 0.74 (contextual 0.77 MRR)
  - [ ] Contextual retrieval at `--size 400` (size-fair re-test)
  - [ ] Step 5: vision captions for figures (Qwen3.6-35B-A3B), tables summary
  - [ ] Embedding model as a variable: nomic-v1.5 vs Qwen3-Embedding-0.6B; chunk sizes 256 / 512 / 1024
  - [ ] Grow `golden.jsonl` to 40 questions (tables + figures, the two-column papers), re-run the ablation
- [ ] 03 ANN benchmark + hybrid search with RRF
- [ ] 04 Query transformation module
- [ ] 05 Reranking + parent-child + context compression
- [ ] 06 Eval harness in CI + tracing
- [ ] 07 Agentic RAG + one advanced paradigm
- [ ] 08 Production hardening (ACL, injection, idempotency, caching)
- [ ] 09 PolicyPal deployed · DocuMind deployed
- [ ] 10 Ablation table + system-design script rehearsed

## Core reading list (whole journey)

- Lewis et al., 2020: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (the original RAG paper)
- Gao et al., 2023/2024: *Retrieval-Augmented Generation for Large Language Models: A Survey*
- Liu et al., 2023: *Lost in the Middle: How Language Models Use Long Contexts*
- Anthropic, *Introducing Contextual Retrieval*: anthropic.com/research/contextual-retrieval
- Malkov & Yashunin, 2016: *HNSW* paper
- Es et al., 2023: *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- Edge et al., 2024: *From Local to Global: A Graph RAG Approach* (Microsoft GraphRAG)
- Asai et al., 2023: *Self-RAG*; Yan et al., 2024: *Corrective RAG (CRAG)*
- Faysse et al., 2024: *ColPali: Efficient Document Retrieval with Vision Language Models*
