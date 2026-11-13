# Model Guide: Local-First with LM Studio, then AWS Bedrock

This plan is **local-first**. You learn every RAG mechanism with free models on your own machine through LM Studio. Once you understand it, you move the same code to Amazon Bedrock (Phase 09) by changing configuration, not rewriting.

This document explains **which model plays which role, why it was chosen, what problem it solves**, how to set it up in LM Studio, and what replaces it on AWS.

---

## 0. Final model checklist (what to download)

### Models in LM Studio (6 total: 2 already installed, 4 to download)

| # | Model (search this name in LM Studio's Discover tab) | Type | Quant | Approx. size | Status | Job | First needed |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen3-30B-A3B-Instruct-2507** | Chat | Q4_K_M | ~18 GB | ✅ Installed | Main model: writes the final answers; also the agent in Phase 07 | Phase 00 |
| 2 | **Qwen3.6-35B-A3B** (keep the `mmproj` file with it) | Chat + vision | Q4_K_M | ~21 GB | ✅ Installed | Captions figures and charts; answers questions from page images | Phase 02 |
| 3 | **Qwen3-4B-Instruct-2507** | Chat | Q4_K_M (or Q8_0) | ~2.5 GB (4.3 GB) | ⬇️ Download | Fast helper: query rewriting, HyDE, a context line per chunk, grading | Phase 00 |
| 4 | **gpt-oss-20b** | Chat (reasoning) | MXFP4 (default) | ~12 GB | ⬇️ Download | Grades answers (judge); second agent to compare against | Phase 06 |
| 5 | **nomic-embed-text-v1.5** | Embedding | F16 / Q8 | ~0.15 GB | ⬇️ Download | First embedding model; teaches query/document prefixes | Phase 00 |
| 6 | **Qwen3-Embedding-0.6B** | Embedding | Q8_0 | ~0.6 GB | ⬇️ Download | Main embedding model from Phase 02 onward | Phase 02 |

**Optional:** **Qwen3-Embedding-4B** (Q4_K_M, ~2.5 GB) for the "bigger embedding model vs smaller" comparison in Phases 02–03.

**Total new downloads:** about 16 GB (19 GB with the optional one).

### Python models (not in LM Studio; they download automatically from Hugging Face)

| Model | Job | Phase |
|---|---|---|
| `all-MiniLM-L6-v2` | Tiny embedding model, used to show what a weak model gets wrong | 01 |
| `BAAI/bge-m3` | Three kinds of search vectors from one model (dense, sparse, multi-vector), for comparison | 03 |
| `BAAI/bge-reranker-v2-m3` | Main reranker (cross-encoder) | 05 |
| `BAAI/bge-reranker-base` | Faster reranker, as a speed baseline | 05 |
| `vidore/colqwen2.5` | Searches page screenshots directly | 07 |

> **Why rerankers aren't in LM Studio:** LM Studio's API serves chat and embedding models, not rerankers. The reranker runs in your Python process on the GPU (`sentence-transformers` `CrossEncoder`), next to LM Studio. See [Phase 05](05-post-retrieval-reranking/README.md).

### Loading settings

- **Qwen3-30B-A3B, Qwen3.6-35B-A3B and gpt-oss-20b:** set the context length to **16,384–32,768**. The default can cut off retrieved text without warning.
- **Qwen3-4B:** a context length of 8,192–16,384 is enough. Put all its layers on the GPU, since it fits.
- **Embedding models:** keep them loaded the whole time; they're small.
- **Qwen3.6-35B-A3B and gpt-oss-20b:** load them only when you need them (captioning, or a grading run), to leave GPU memory for the main model.

After downloading, run `lms ls` (or open `http://localhost:1234/v1/models` with the server running). Copy the exact model names it shows into `GEN_MODEL`, `HELPER_MODEL`, `VISION_MODEL`, `JUDGE_MODEL` and `EMBED_MODEL` in your `.env` (see §6).

---

## 1. Why local first?

| Benefit | Why it matters for learning |
|---------|-----------------------------|
| Free, unlimited iteration | Re-run ingestion and 100-question evals as often as you like. No API bill. |
| No API key needed | Start today. |
| Privacy | Use real company or personal documents without sending them anywhere. |
| You see the raw mechanics | Context length limits, tokenization, quantization and latency are visible, not hidden behind an API. |
| Weaker models expose weak retrieval | A frontier model can sometimes guess around bad retrieval. A local model can't, so your eval shows retrieval problems clearly. That makes you a better RAG engineer. |

Trade-offs to keep in mind: local models are slower and less capable than frontier cloud models, and local "cost" is **time** instead of money. So you'll record **latency and token counts** for every run, then convert them to an estimated cloud cost before migrating (Phase 09).

---

## 2. Your hardware and what it means

Detected on this machine:

| Component | Spec | What it means |
|-----------|------|---------------|
| GPU | NVIDIA RTX 5080, **16 GB VRAM** | Dense models up to ~14B (Q4) fit fully on the GPU. Bigger models split between GPU and RAM. |
| RAM | **127 GB** | Very large models can live partly in system RAM. Ideal for **Mixture-of-Experts (MoE)** models. |
| CPU | Intel Core Ultra 9 285K | Fast enough for CPU-side expert layers, embeddings and BM25. |

**Rule of thumb for GGUF size:** at Q4_K_M quantization, a model needs about **0.6 GB per billion parameters**, plus memory for the KV cache (which grows with context length).

**Why MoE models suit you:** a model like Qwen3-30B-A3B has 30B parameters in total, but only ~3B are **active** for each token. LM Studio can keep the shared layers on the GPU and the expert weights in your large RAM. You get the knowledge of a 30B model at close to the speed of a 3B model.

### Quantization in one minute

| Quant | Size vs FP16 | Quality | Use |
|-------|--------------|---------|-----|
| Q8_0 | ~50% | ≈ original | Small models (embeddings, 4B helpers) |
| Q6_K | ~40% | very close | When it fits |
| **Q4_K_M** | ~30% | small loss | The default for 20–35B models |
| Q3 / Q2 | <25% | noticeable loss | Avoid for RAG (instruction following degrades) |

---

## 3. The model lineup (role → model → why)

A real RAG system uses **several models**, each for a different job. Here is your lineup.

### 3.1 Generator: writes the final grounded answer

**✅ Qwen3-30B-A3B-Instruct-2507 (Q4_K_M), already downloaded**

| Feature | Why it matters for RAG |
|---------|------------------------|
| MoE: 30.5B total, ~3.3B active per token | Fast on your GPU + RAM split, with a big model's knowledge |
| **Non-thinking "Instruct-2507" variant** | Answers directly, with no `<think>` blocks. Lower latency and clean output for citations. |
| Long native context (256K tokens) | Room for many retrieved chunks. You'll set 16–32K in LM Studio to save memory. |
| Strong instruction following | Obeys "answer only from the sources, cite [n], otherwise say you couldn't find it" |
| Multilingual (incl. Hindi) | Useful if you test non-English questions |
| Tool / function calling | The same model works as the agent in Phase 07 |

**Problems it solves:** grounded generation, citation formatting, refusing when the answer is missing, and agentic tool use, all at usable local speed.

### 3.2 Vision + harder reasoning

**✅ Qwen3.6-35B-A3B (Q4_K_M + `mmproj` file), already downloaded**

- The `mmproj-...gguf` file next to it is a **vision projector**. In LM Studio this model accepts **images**, not just text.
- Also MoE (~3B active), so it's fast despite its size.
- **Use it for:** figure and chart captioning at ingestion (Phase 02), answering from page screenshots (Phase 07 multimodal RAG), and as a stronger second opinion on hard questions.
- Check its model card in LM Studio for whether it emits thinking (`<think>...</think>`) blocks. The `llm.py` helper below strips them either way.

**Problems it solves:** charts, diagrams, scanned pages and tables whose meaning is in the image. Text-only parsing loses all of that.

### 3.3 Fast helper: the many small calls

**⬇️ Qwen3-4B-Instruct-2507 (Q4_K_M or Q8_0), to download** (~2.5–4.5 GB)

- Fits **fully in VRAM** next to an embedding model, so it's very fast.
- **Use it for** the high-volume, simple calls: query rewriting, multi-query and HyDE (Phase 04), contextual-retrieval sentences for every chunk (Phase 02), chunk relevance grading in Corrective RAG (Phase 07).

**Problem it solves:** contextual retrieval needs one LLM call **per chunk** (thousands of calls). With the 30B model that's slow; with a 4B model it's practical. This mirrors production, where helper calls go to a small, cheap model.

### 3.4 Judge: grades answers in evaluation

**⬇️ gpt-oss-20b (OpenAI open-weight, MoE), to download** (~12–13 GB)

- **A different model family** from your Qwen generator. LLM judges tend to favour answers from their own family (self-preference bias), so judging Qwen with a non-Qwen model is more honest.
- A reasoning model with adjustable reasoning effort and good structured (JSON) output, which suits rubric-based grading.
- Fits on the 16 GB GPU.
- Alternatives if you prefer: **Gemma 3 27B** (Google) or **Mistral Small 3.2 24B**.

**Problem it solves:** trustworthy faithfulness and correctness scores in Phase 06, after you calibrate them against your own labels.

### 3.5 Embedding models: turn text into vectors

You'll compare these on your golden set. Choosing an embedding model **by evidence** is a core skill.

| Model | Where it runs | Dims | Max input | Key features | Why it's in the plan |
|-------|---------------|------|-----------|--------------|----------------------|
| **nomic-embed-text-v1.5** ⬇️ | LM Studio (`/v1/embeddings`) | 768 (Matryoshka: down to 64) | 8,192 tokens | Small (137M), fast, long input, **needs task prefixes** (`search_query:` / `search_document:`) | **Phase 01 starting model.** Easiest in LM Studio. Teaches query/document prefixes and Matryoshka truncation (Phase 03). |
| **Qwen3-Embedding-0.6B** ⬇️ (try 4B later) | LM Studio GGUF or `sentence-transformers` | up to 1,024 (Matryoshka) | 32K tokens | Top-tier retrieval quality for its size, 100+ languages, **instruction-aware queries** | **Main embedding model from Phase 02.** Same family as your generator. The 0.6B vs 4B comparison teaches the size/quality trade-off. |
| **BAAI/bge-m3** | Python (`FlagEmbedding` / `sentence-transformers`) on the GPU | 1,024 | 8,192 tokens | One model outputs **dense + sparse (lexical weights) + multi-vector (ColBERT)** representations; multilingual | **Phase 03 and 05:** learn learned-sparse retrieval and late interaction with a single model. Strong on Hindi and mixed-language text. |
| **BAAI/bge-small-en-v1.5** or **all-MiniLM-L6-v2** | Python, even on CPU | 384 | 512 / 256 tokens | Tiny and fast; English only; short input limit | **Lab 1a baseline.** Shows how a weak model and a short input limit fail (silent truncation), so you appreciate the better ones. |

> **Never mix embedding models in one index.** Each model has its own vector space. Switching models (including local → Bedrock Titan) means **re-embedding the whole corpus** into a new index.

### 3.6 Rerankers (cross-encoders)

LM Studio serves chat and embedding endpoints, **not** reranking, so rerankers run in Python on your GPU with `sentence-transformers`.

| Model | Size | Why |
|-------|------|-----|
| **BAAI/bge-reranker-v2-m3** | 568M | **Default.** Strong, multilingual, pairs with bge-m3. Fast on an RTX 5080 for 50 candidates. |
| BAAI/bge-reranker-base | 278M | A faster baseline for the "rerank vs latency" ablation |
| Qwen3-Reranker-0.6B (optional) | 0.6B | Same family as your embeddings; LLM-based reranker. Needs `transformers` usage per its model card. |

**Problem they solve:** embeddings retrieve 50 *roughly* relevant chunks; a cross-encoder reads query and chunk together and picks the 5 that really answer the question (Phase 05).

### 3.7 Visual document retrieval (Phase 07)

| Model | Why |
|-------|-----|
| **vidore/colqwen2.5** (or `vidore/colpali`) via `colpali-engine` / `byaldi` | Embeds **page screenshots** directly (no OCR or parsing). Great for slides, charts and complex layouts. Runs on your GPU. |

---

## 4. What to load when (VRAM budget per phase)

LM Studio can keep several models loaded (and can load on demand: "JIT loading"). A sensible setup per phase:

| Phase | Load in LM Studio | Python-side (GPU) |
|-------|-------------------|-------------------|
| 00–01 | Qwen3-30B-A3B-Instruct + nomic-embed | bge-small (Lab 1a) |
| 02 | + Qwen3-4B helper (contextual retrieval), Qwen3.6-35B-A3B for figure captions (load only when captioning) | Qwen3-Embedding-0.6B |
| 03 | Embedding models only (ingestion) | FAISS, bge-m3 |
| 04 | Qwen3-30B-A3B + Qwen3-4B | — |
| 05 | Qwen3-30B-A3B | bge-reranker-v2-m3 |
| 06 | Qwen3-30B-A3B (generator) + gpt-oss-20b (judge); run them **one after the other** if VRAM is tight | — |
| 07 | Qwen3-30B-A3B (agent, tools), Qwen3.6-35B-A3B (vision) | colqwen2.5 |

If memory runs short: unload what you don't need, lower the context length, or let LM Studio put more layers or experts on the CPU.

---

## 5. LM Studio setup (step by step)

1. **Open the LM Studio app once.** Your `lms` CLI currently reports "no running LM Studio", so the app (or its server) must be running for the CLI and the API to work. If `lms` isn't on your PATH, run `~/.lmstudio/bin/lms bootstrap` once.
2. **Download the extra models** (Discover tab, search by name): `Qwen3-4B-Instruct-2507`, `gpt-oss-20b`, `nomic-embed-text-v1.5`, `Qwen3-Embedding-0.6B` (GGUF).
3. **Start the local server:** Developer tab → Start Server, or `lms server start`. It serves an **OpenAI-compatible API** at `http://localhost:1234/v1`.
4. **Load the generator with a real context length.** ⚠️ The default context length is often small (e.g. 4,096 tokens). Retrieved chunks then get **silently truncated**. Set **16,384–32,768**:
   ```bash
   lms ls                                   # see exact model identifiers
   lms load <qwen3-30b-a3b-instruct-2507 identifier> --context-length 32768
   ```
   In the app: model settings → Context Length, GPU Offload, and (in recent versions) the option to keep MoE expert weights on the CPU. That option lets 30B MoE models run fast on 16 GB of VRAM.
5. **Check it works:**
   ```bash
   curl http://localhost:1234/v1/models
   ```
6. **Python packages:**
   Use **uv** with Python 3.12, and configure the CUDA 12.8 PyTorch index in `pyproject.toml` first. The full steps are in [Phase 00, Step 1](00-prerequisites/README.md#step-1-python-environment).
   ```bash
   uv add torch                     # from the cu128 index configured in pyproject.toml (RTX 50-series needs it)
   uv add openai numpy python-dotenv sentence-transformers transformers
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

What LM Studio's API gives you (all OpenAI-compatible, so you use the `openai` Python package):
- `POST /v1/chat/completions`: chat, **streaming**, **tool calling** (Qwen3, gpt-oss), **structured output** (`response_format` with a JSON schema), and **images** for vision models
- `POST /v1/embeddings`: embedding models
- `GET /v1/models`: what's available
- The `usage` field returns prompt and completion token counts, so you can track tokens exactly as you would on a paid API.

---

## 6. One code base, two providers

All phases use two small wrappers, `rag/llm.py` and `rag/embed.py` (written in Phase 01). They read the provider from `.env`:

```ini
# .env: local (Phases 00–08)
LLM_PROVIDER=lmstudio
LMSTUDIO_URL=http://localhost:1234/v1
GEN_MODEL=qwen3-30b-a3b-instruct-2507          # use the exact identifier from `lms ls`
HELPER_MODEL=qwen3-4b-instruct-2507
VISION_MODEL=qwen3.6-35b-a3b
JUDGE_MODEL=openai/gpt-oss-20b
EMBED_PROVIDER=lmstudio
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
```

```ini
# .env: AWS (Phase 09). Same code, different config
LLM_PROVIDER=bedrock
AWS_REGION=us-west-2
GEN_MODEL=<Bedrock model or inference-profile ID from the console>
HELPER_MODEL=<small Bedrock model ID>
EMBED_PROVIDER=bedrock
EMBED_MODEL=amazon.titan-embed-text-v2:0
```

---

## 7. Local → AWS Bedrock mapping (Phase 09)

| Role | Local (learning) | AWS (production) | Notes |
|------|------------------|------------------|-------|
| Generator | Qwen3-30B-A3B-Instruct-2507 | Claude (Sonnet or Opus tier) on Bedrock; Qwen3 or other open-weight models on Bedrock if offered in your region | Check model availability per region in the Bedrock console |
| Helper | Qwen3-4B-Instruct-2507 | Claude Haiku tier or Amazon Nova Micro/Lite | Small, cheap, fast |
| Vision | Qwen3.6-35B-A3B (+mmproj) | Claude (vision) or Amazon Nova Lite/Pro | Same image-in, text-out pattern |
| Judge | gpt-oss-20b | A model from a different family than your generator; or Bedrock Evaluations | Keep the judge ≠ generator rule |
| Embeddings | nomic / Qwen3-Embedding / bge-m3 | Titan Text Embeddings V2; Cohere Embed (multilingual); Cohere Embed v4 / Nova Multimodal for images | **Must re-embed everything** |
| Reranker | bge-reranker-v2-m3 | Amazon Rerank 1.0, Cohere Rerank 3.5 (Bedrock Rerank API) | Check region availability |
| Vector store | Chroma / Qdrant / OpenSearch in Docker | OpenSearch, Aurora pgvector, S3 Vectors | Docker OpenSearch = same API as AWS |
| Parsing | PyMuPDF, Docling | Textract, Bedrock KB FM parser | |
| Tracing | Phoenix / Langfuse (local) | CloudWatch, X-Ray, Langfuse | |

**Migration checklist (Phase 09):**
1. Switch `.env` to Bedrock; run unit tests.
2. Re-embed the corpus into a **new** index with the Bedrock embedding model.
3. Run the **same golden set**; compare with your local numbers in `results.md`.
4. Re-tune prompts if needed (stronger models may need *less* instruction).
5. Compare latency and **real cost per query** against the estimates you made from local token counts.

---

## 8. Sampling settings for RAG

- **Generator:** temperature 0.1–0.3 for grounded answers (lower = more literal). Qwen's general-chat recommendation is higher (~0.7), but RAG wants faithfulness over creativity.
- **Helpers (rewrite, grading, JSON):** temperature 0, with JSON-schema structured output.
- **Judge:** temperature 0; use reasoning if it's a reasoning model.
- On Bedrock, some newer models manage sampling internally and reject a `temperature` parameter. The `llm.py` wrapper only sends it to Bedrock when you set it explicitly.
