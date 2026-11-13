# Phase 09: Capstone Projects on AWS

**Goal:** Build two deployable, portfolio-grade RAG systems: one on the **managed** path (fast, low-ops) and one **custom** (every component by hand). Then you can defend either choice to a customer or an interviewer.
**Time:** 3–4 weeks part-time (PolicyPal: 1 weekend–1 week; DocuMind: 2–3 weeks).
**Guide sections:** **Part 3** has the full designs, decision tables, code and build steps. This README is the learning plan around them.

---

## Why this phase matters

Phases 01–08 taught each component on your laptop. Employers want to see that you can **ship** one on a cloud, with auth, infrastructure as code, evaluation and cost control, and that you can explain *why* you chose managed vs custom.

| | Project 1: PolicyPal | Project 2: DocuMind |
|---|---|---|
| Use case | Employee Q&A over HR and company policies | Multi-tenant Q&A over manuals and financial reports with tables, charts and product codes |
| Pipeline | Managed: Bedrock Knowledge Bases | Custom: Step Functions + your code |
| Vector store | S3 Vectors | OpenSearch (managed domain; Docker locally) |
| Search | Semantic + Bedrock reranker | Hybrid (k-NN + BM25) + RRF + reranker + parent expansion |
| Compute | Lambda + API Gateway | Step Functions + Lambda (ingestion); ECS Fargate FastAPI (query, streaming) |
| You learn | KB config, metadata filtering, guardrails, Bedrock Evaluations, IaC | Every RAG internal, idempotent ingestion, multi-tenancy, eval-driven tuning, CI gates |

**Before you start:** set an **AWS Budgets alert** (e.g. $10–20), work in one region (the guide uses `us-west-2` because the rerankers are available there), and tag every resource so you can find and delete it.

---

## Step 0: AWS fundamentals you need (if new to AWS)

- IAM: users vs roles, least-privilege policies, `AWS_PROFILE`
- S3 (versioning, SSE-KMS), Lambda, API Gateway HTTP API, Cognito (user pools, groups, JWT)
- CloudWatch Logs and metrics, X-Ray
- Infrastructure as code: **AWS CDK** (Python or TypeScript) or SAM/CloudFormation
- Amazon Bedrock: model access, `Converse` / `ConverseStream`, Knowledge Bases, Guardrails, Rerank, Evaluations
- For DocuMind: Step Functions, SQS, EventBridge, DynamoDB, Textract, ECS Fargate, ECR, ALB, VPC endpoints

Resources: AWS Skill Builder (*Amazon Bedrock Getting Started*), the Bedrock User Guide, the CDK Workshop.

---

## Step 1: Migrate your local pipeline to Bedrock (before the projects)

Until now everything ran on LM Studio. Do this migration **first**, on your own `rag-lab` code. It's the most valuable comparison in the whole plan: *same pipeline, same golden set, local vs cloud models.* Full mapping table: [MODELS.md §7](../MODELS.md#7-local--aws-bedrock-mapping-phase-09).

| Role | Local (Phases 00–08) | Bedrock (now) |
|------|----------------------|---------------|
| Generator | Qwen3-30B-A3B-Instruct-2507 | A Claude model on Bedrock (or an open-weight Qwen model if offered in your region) |
| Helper | Qwen3-4B-Instruct-2507 | A small, fast model (Claude Haiku tier, Amazon Nova Micro/Lite) |
| Vision | Qwen3.6-35B-A3B | Claude (vision) or Amazon Nova Lite/Pro |
| Judge | gpt-oss-20b | A different family than your generator, or Bedrock Evaluations |
| Embeddings | Qwen3-Embedding-0.6B / nomic | Titan Text Embeddings V2 (1,024 dims) or Cohere Embed |
| Reranker | bge-reranker-v2-m3 | Bedrock Rerank (Amazon Rerank 1.0 / Cohere Rerank 3.5) |

Steps:
1. AWS account basics (below), a budget alert, and Bedrock **model access** for the models above in your chosen region.
2. Configure credentials (`aws configure sso` or a named profile), then switch `.env`:
   ```ini
   LLM_PROVIDER=bedrock
   EMBED_PROVIDER=bedrock
   AWS_REGION=us-west-2
   GEN_MODEL=<model or inference-profile ID from the Bedrock console>
   HELPER_MODEL=<small model ID>
   EMBED_MODEL=amazon.titan-embed-text-v2:0
   ```
   Your `rag/llm.py` and `rag/embed.py` already contain the Bedrock branches (Converse API and Titan `invoke_model`).
3. Add a Bedrock branch to `rag/rerank.py` using the Bedrock `rerank` API (`bedrock-agent-runtime`; see Guide Part 3, `retrieval.py`).
4. **Re-embed the corpus into a new index.** Local vectors and Titan vectors are not comparable.
5. Run the **same golden set**. Fill in:

| Config | hit@5 | MRR | Faithfulness | p95 latency | Cost / 1k queries |
|--------|-------|-----|--------------|-------------|-------------------|
| Local: Qwen3-Embedding + bge-reranker + Qwen3-30B | | | | | ~$0 (time only) |
| Bedrock embeddings + local generator | | | | | |
| Full Bedrock | | | | | |

6. Compare actual cost with the **estimate** you made from local token counts in Phase 06. Note where prompts needed changes (stronger models often need fewer instructions).

This table answers a question customers often ask: *"Can we run this on open models on our own hardware, or do we need a cloud LLM?"*

---

## Project 1: PolicyPal (managed RAG on Bedrock Knowledge Bases)

**Goal:** Employees sign in and ask "How many sick days do L2 engineers get?". PolicyPal answers only from policy PDFs, cites document and page, says when it doesn't know, and never shows managers-only policies to other employees.

### Build steps (follow Guide Part 3, Project 1)

1. **Model access + budget alert** in Bedrock (`us-west-2`): Titan Text Embeddings V2, a Claude model, Amazon Rerank 1.0.
2. **Documents bucket** (versioning + KMS) with 10–20 policy PDFs, each with a `.metadata.json` sidecar (`audience`, `department`, `doc_year`).
3. **Create the Knowledge Base in the console first:** FM parser (for tables), hierarchical chunking (parent ≈1,500 / child ≈300 tokens), Titan V2 at 1,024 dims, S3 Vectors. Test in the console.
4. **Guardrail:** PII filters + contextual grounding check (start grounding ≈0.75, relevance ≈0.5).
5. **Ask Lambda:** `Retrieve` (25 candidates, audience filter from **Cognito groups**, reranking to 5) → `Converse` with a strict cite-or-refuse system prompt → `ApplyGuardrail` grounding check. API Gateway with a JWT authorizer.
6. **Auto-sync:** S3 events → EventBridge → SQS (batching window) → sync Lambda (`StartIngestionJob`, retry on conflict).
7. **Frontend:** React on S3 + CloudFront; Cognito login; citations rendered as links with page numbers.
8. **Evaluate:** 40-question golden set (labelled by doc + page); retrieval script (hit@5, MRR) + a Bedrock Evaluations job (correctness, completeness, faithfulness).
9. **Codify** everything in CDK so you can `cdk destroy` and recreate it.

### Learning outcomes to note in your README
- Why Knowledge Bases (vs custom), why S3 Vectors (no idle cost; generation dominates latency), why the FM parser (tables), why hierarchical chunking
- How the access-control filter is built server-side from verified groups
- Your eval numbers, and what you'd change if hybrid search became necessary

### Tests
- Unit tests with `botocore.stub.Stubber` (group parsing, empty results, not-found path, guardrail-blocked path)
- **Security test:** a non-manager asks about a managers-only policy; the citations must never include it

### Cleanup (important)
Delete the KB, data source, vector index and bucket, guardrail and CloudFront distribution, or run `cdk destroy`. Note: a Quick-create OpenSearch Serverless collection is **not** deleted with the KB and bills while idle.

---

## Project 2: DocuMind (custom multimodal hybrid RAG)

**Goal:** Several tenants upload technical manuals and quarterly reports. Users ask "What torque does part HX-4410 need?" or "How did Q3 operating margin compare with Q2?". Answers handle product codes, tables and charts, stream to the UI, cite pages and **never mix tenants' data**.

This project is your Phases 02–08 code, deployed. Reuse your `rag/` modules wherever you can.

### Build order (follow Guide Part 3, Project 2)

**Week 1: local first**
1. `docker-compose.yml`: OpenSearch + Dashboards + your FastAPI app (`AUTH_MODE=local`, two fake tenants). Develop against **LM Studio models first** (`LLM_PROVIDER=lmstudio`, `LMSTUDIO_URL=http://host.docker.internal:1234/v1`), then flip to Bedrock. Keep the two in **separate indexes**, because their embeddings aren't comparable.
2. Index mappings: `knn_vector` (HNSW, faiss, inner product, M=16, ef_construction=128) + BM25 `text` + keyword metadata; a separate `parents` index.
3. Ingestion library: `LocalParser` (PyMuPDF) and `TextractParser` behind one interface → structure-aware parent-child chunker (tables row-split with headers) → enrichment (figure captions + contextual sentences) → embed → `replace_document` (delete-by-doc then bulk upsert with deterministic IDs).
4. Query path: rewrite → **k-NN and BM25 in parallel under a mandatory tenant/groups filter** → RRF → rerank → parent expansion → `ConverseStream` over SSE → grounding check.
5. Tests: `split_table`, `chunk_document`, `rrf` units; integration against Docker OpenSearch; **tenant-isolation test**.

**Week 2: cloud ingestion**
6. S3 → EventBridge → SQS → Step Functions: Register (hash check, version) → Parse (async Textract with Wait state) → Chunk (write to S3; pass keys, not text) → Enrich (Map state, MaxConcurrency ≈5) → Embed & Index (Retry on throttling with backoff + jitter) → Mark indexed / Mark failed. A DynamoDB registry. Delete flow on `ObjectRemoved`.
7. DLQ + CloudWatch alarm on failed executions.

**Week 3: cloud query, CI, evaluation**
8. API: ECR image → ECS Fargate behind an ALB; JWT verification against Cognito JWKS; autoscaling. VPC endpoints for S3, Bedrock, DynamoDB and Logs.
9. Frontend: React; read the SSE stream with `fetch` + a stream reader (so the Authorization header can be sent).
10. **CI/CD:** on each PR, run unit tests, build the image, run the golden-set eval against a staging index, and **fail on regression** (`--min-hit-at-5 0.85 --min-mrr 0.70`).

### The evaluation table (your strongest interview story)

Run the same 60–100-question golden set (tables, figures, product codes, unanswerable, multi-hop) through each configuration:

| Configuration | hit@5 | MRR | Faithfulness | p95 latency | Cost / 1k queries |
|---------------|-------|-----|--------------|-------------|-------------------|
| Vector only | | | | | |
| + BM25 (hybrid, RRF) | | | | | |
| + Reranker | | | | | |
| + Contextual retrieval | | | | | |
| + Parent expansion | | | | | |

Fill it with **your own numbers**, then write down which technique helped which question type, and which one you'd drop because it didn't pay for itself.

### Stretch goals (pick 1–2)
- Agentic RAG: expose `search_documents(query, filters)` as an MCP tool; agent-driven multi-hop comparison questions (Phase 07)
- Tiered storage: recent documents in OpenSearch, archive in S3 Vectors
- Semantic cache per tenant; thumbs-down → golden-set candidates
- Multilingual: Cohere multilingual embeddings + Hindi test questions
- ColPali page retrieval for chart-heavy reports

---

## Portfolio packaging (do this for both projects)

Each project repo should have:
- **README:** problem, architecture diagram, design-decision table ("chose X because…, not Y because…"), eval table, cost estimate, how to run locally, how to deploy, how to tear down
- A 2–3-minute **demo video** (login → question → streaming answer with citations → the access-control test)
- `eval/` with the golden set format (not confidential data) and the latest report
- Clean IaC; no secrets in git
- A short **blog post or LinkedIn write-up** on one learning (e.g. "What hybrid search fixed in my RAG, with numbers")

## Milestone

- PolicyPal deployed, evaluated, documented, torn down cleanly (IaC can recreate it)
- DocuMind running locally with Docker *and* deployed on AWS, with a CI eval gate and a filled evaluation table
- Both repos public (with non-confidential sample data) and linked from your CV

## Exit checklist

- [ ] I can explain managed (Knowledge Bases) vs custom and when I'd choose each, including the middle path (KB + a custom chunking Lambda).
- [ ] I can draw both architectures from memory.
- [ ] I can justify every row of both design-decision tables.
- [ ] I know the cost drivers of each project and how I kept them low.
- [ ] I have my own numbers for what each retrieval technique added.

## Next

→ [Phase 10: Interview Prep & Career](../10-interview-prep/README.md)
