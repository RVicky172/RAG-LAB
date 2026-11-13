# Phase 10: Interview Prep & Career

**Goal:** Explain and defend every RAG decision clearly, under time pressure, with evidence from your own projects.
**Time:** ongoing; intensive in the 1–2 weeks before interviews.
**Guide sections:** **Part 4** has 30 questions with model answers and feedback, the system-design script, red flags and the cheat sheet. Use it as your question bank. This README is the practice plan.

---

## Why this phase matters

Knowing RAG and *communicating* RAG are different skills. Interviewers listen for: trade-offs instead of single answers, evaluation instead of vibes, security in retrieval instead of the prompt, and real numbers from things you built.

---

## Learning steps (in order)

### Step 1: Build your story bank (from your own work)

Write 6 short stories in STAR form (Situation, Task, Action, Result), each with a number from your `results.md`:

1. **A retrieval improvement:** e.g. "Hybrid search raised hit@5 on identifier questions from 0.40 to 0.90."
2. **A chunking decision:** why structure-aware + contextual beat semantic on your corpus.
3. **A failure you debugged** stage by stage using traces.
4. **A security design:** tenant isolation with pre-filters and isolation tests.
5. **A cost or latency win:** reranking to fewer chunks, caching, or skipping rewrites.
6. **A technique you dropped** because evaluation showed it didn't pay for itself. (This one impresses the most.)

### Step 2: Master the 30 core questions

Practise Guide Part 4's questions **out loud, under 1 minute each**, recording yourself. Categories:

| Category | Questions | Phase |
|----------|-----------|-------|
| Fundamentals | Q1–Q3 | 01 |
| Parsing & chunking | Q4–Q8 | 02 |
| Embeddings | Q9–Q11 | 01, 03 |
| Vector search | Q12–Q16 | 03, 05 |
| Production | Q17–Q21 | 08 |
| Evaluation | Q22–Q25 | 06 |
| Scenarios | Q26–Q30 | 06–09 |

Add these questions from the enhanced plan (not in the guide):
- What is HyDE and when does it hurt? (Phase 04)
- Explain "Lost in the Middle" and how you order context. (Phase 05)
- What is late chunking vs contextual retrieval? (Phase 02)
- When does GraphRAG beat vector RAG? What does it cost? (Phase 07)
- CRAG vs Self-RAG? (Phase 07)
- How would you handle charts in scanned reports? (Phase 07: captions vs ColPali)
- Long context with 1M tokens: is RAG dead? (Phase 01: cost, latency, freshness, access control, citations)

### Step 3: The system-design script (5–7 minutes)

Rehearse this structure until it's automatic:

1. **Clarify requirements:** users, query volume, latency target, document types and volume, update frequency, languages, access control, residency.
2. **Draw two pipelines:** ingestion (offline) and query (online), sharing the vector store.
3. **Ingestion choices:** parser per document type, chunking + metadata, enrichment, embedding model, idempotent updates.
4. **Storage choice:** justified by scale, latency, hybrid needs, filtering, existing stack.
5. **Query path:** rewrite → mandatory filters → hybrid + RRF → rerank → context assembly → generation with citations → guardrails → streaming.
6. **Evaluation and observability:** golden set, metrics, CI gate, traces, online feedback.
7. **Operations:** security, cost levers, failure modes, what you'd build first vs later.

Practice prompts:
- RAG assistant over 5 million support tickets for 300 customers (Guide Q28)
- Internal code-search assistant over 2,000 repositories
- Legal contract Q&A with strict citations and audit logging
- Customer-facing product assistant with 50K QPS peaks
- Financial-report analyst over 10 years of PDFs with charts

### Step 4: Red flags to avoid

- Choosing a chunk size or vector DB without explaining why
- Relying on the LLM prompt for access control
- Ignoring document updates and deletions
- No evaluation plan ("we'll test it manually")
- Mixing embedding models, or forgetting to re-embed after a model change
- Confusing MRR with MMR, or recall with precision
- "RAG is always the answer" / "Agents for everything"

### Step 5: Cheat sheet (memorize)

| Topic | Remember |
|-------|----------|
| Starting chunk size | 300–800 tokens, 10–20% overlap; tune on a golden set |
| Candidates → reranked | Retrieve 20–50, rerank to 3–8 |
| RRF | score = Σ 1 / (60 + rank); rank-based, no score normalization |
| Recall@k / MRR | Found at all? / How high? Low recall → chunking/search; low MRR → rerank |
| Contextual retrieval | Anthropic benchmark: ~49% fewer retrieval failures; ~67% with reranking |
| HNSW knobs | M, ef_construction (build), ef_search (query recall vs latency) |
| Normalized vectors | cosine = dot product ranking = L2 ranking |
| Pre- vs post-filter | Pre keeps k results and is required for security |
| Faithfulness | Every claim supported by the retrieved context |
| Golden set labels | Document + page span, never chunk IDs |

### Step 6: Keep learning (stay current)

RAG moves fast. Build a weekly 2-hour habit:
- Read 1 paper or engineering blog post (Anthropic, LlamaIndex, Qdrant, Weaviate, Pinecone, and the AWS ML blog)
- Re-run your eval when a new embedding model or reranker appears; post the result
- Follow the MTEB leaderboard and model release notes
- Contribute: a small PR to an open-source RAG tool (RAGAS, Docling, Qdrant client) is a strong signal

### Step 7: Career assets checklist

- [ ] CV bullets with numbers: "Built a multi-tenant hybrid RAG on AWS (OpenSearch, Bedrock); raised hit@5 from 0.62 to 0.91 with BM25 + reranking; CI eval gate."
- [ ] Two public project repos with READMEs, architecture diagrams, eval tables and demo videos
- [ ] 2–3 write-ups (blog or LinkedIn) on lessons with numbers
- [ ] GitHub profile README linking the projects
- [ ] Mock interviews: 3 system-design and 2 deep-dive sessions with a peer (or record yourself)

---

## Milestone

- Story bank (6 STAR stories with numbers)
- All 30 + 7 questions answered out loud in under a minute each, recorded at least once
- 3 system designs drawn and timed at 7 minutes or less
- CV and portfolio updated

## You've finished the roadmap when you can…

- Build a RAG system from raw parts **and** with a framework, and explain what the framework does underneath
- Choose parsers, chunkers, embeddings, indexes, retrievers and rerankers **from evidence**, not habit
- Prove quality with an automated evaluation, and debug any failure stage by stage
- Ship it securely on a cloud with access control, injection defense, observability and cost control
- Explain all of it to an engineer, an interviewer, and a business stakeholder

→ Back to the [roadmap](../../README.md)
