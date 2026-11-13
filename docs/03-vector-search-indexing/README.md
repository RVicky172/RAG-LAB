# Phase 03: Vector Databases & Indexing Mechanics

**Goal:** Understand how search scales from thousands to billions of vectors with low latency, and build hybrid (dense + sparse) retrieval.
**Time:** about 2 weeks part-time.
**Guide sections:** Part 1 §6 (Vector databases and vector search); Interview Q11–Q16.

**Models this phase** (see [MODELS.md](../MODELS.md)):

| Role | Model | Why here |
|------|-------|----------|
| Dense embeddings | Qwen3-Embedding-0.6B (1,024 dims, Matryoshka) | Your main model; truncate to 256/512 dims to measure the memory vs recall trade-off |
| Matryoshka comparison | nomic-embed-text-v1.5 (768 → 512/256/128/64) | Trained for truncation; shows how quality degrades with fewer dimensions |
| Learned sparse + multi-vector | **BAAI/bge-m3** (Python, GPU) | One model gives dense, sparse (SPLADE-like lexical weights) and ColBERT vectors, so you can compare all three retrieval styles on one corpus |
| Fine-tuning base (optional) | bge-small-en-v1.5 or Qwen3-Embedding-0.6B | Small enough to fine-tune on an RTX 5080 in minutes |

---

## Why this phase matters

Your NumPy `vectors @ q` is **exact k-NN**: it compares the query with every vector. That's perfect at 10,000 chunks and impossible at 100 million. Production systems use **approximate** nearest-neighbour (ANN) indexes and trade a little recall for big gains in speed. You also saw in Phase 01 that dense vectors miss product codes and IDs. **Hybrid search** fixes that, and it's the single most common production upgrade.

---

## Learning steps (in order)

### Step 1: Exact vs approximate search

| | Exact k-NN (brute force) | ANN |
|---|---|---|
| Accuracy | 100% recall | ~95–99% recall (tunable) |
| Cost per query | O(N × d) | sub-linear |
| Use when | < ~100K–500K vectors, or as ground truth | everything larger |

**Recall@k for ANN** = (overlap between ANN top-k and exact top-k) / k. You always measure an ANN index against brute force.

### Step 2: ANN algorithms

#### HNSW (Hierarchical Navigable Small World)
- A multi-layer graph. Top layers are sparse "highways"; the bottom layer has every vector. Search starts at the top, greedily hops toward the query, and descends layer by layer.
- **Knobs:**
  - `M`: links per node. Higher means better recall and more memory.
  - `ef_construction`: candidate breadth at build time. Higher means a better graph and slower builds.
  - `ef_search`: candidate breadth at query time. **The main recall vs latency dial.**
- Pros: fast, high recall, supports incremental inserts. Cons: memory-hungry (the vectors plus the graph live in RAM); deletes are awkward.
- The default in Qdrant, Weaviate, pgvector, OpenSearch and Milvus.

#### IVF (Inverted File Index)
- k-means clusters the vectors into `nlist` cells (Voronoi cells). A query checks only the `nprobe` nearest cells.
- Knobs: `nlist` (≈ √N is a common start), `nprobe` (recall vs speed).
- Needs a training step; less memory than HNSW; recall drops for vectors near cell borders.

#### PQ (Product Quantization) and other compression
- **PQ:** split each vector into m sub-vectors; replace each with the ID of its nearest centroid (1 byte). A 1,024-dim float32 vector (4 KB) becomes ~64 bytes.
- **Scalar quantization (int8):** 4× smaller, tiny recall loss.
- **Binary quantization:** 1 bit per dimension, 32× smaller. Use it with **rescoring**: retrieve top-k×10 by binary, re-rank with full vectors.
- **IVF-PQ** combines both for billion-scale indexes. **DiskANN** keeps the graph on SSD.
- **Matryoshka embeddings:** models trained so the first 256 dims are useful alone. Truncate to shrink the index; re-check recall.

> **Rule:** after any compression or truncation, re-measure recall against brute force *and* your golden set.

#### Lab 3a: ANN benchmark

Use FAISS on 100K–1M vectors (embed a public dataset or generate them):

```python
import faiss, numpy as np, time

d = 384
xb = np.random.rand(200_000, d).astype("float32"); faiss.normalize_L2(xb)
xq = np.random.rand(1_000, d).astype("float32");   faiss.normalize_L2(xq)

flat = faiss.IndexFlatIP(d); flat.add(xb)
_, gt = flat.search(xq, 10)                                   # ground truth

def bench(index, name):
    t = time.perf_counter(); _, I = index.search(xq, 10); ms = (time.perf_counter() - t) * 1000 / len(xq)
    recall = np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(len(xq))])
    print(f"{name:28s} recall@10={recall:.3f}  {ms:.3f} ms/query")

hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 128; hnsw.add(xb)
for ef in (16, 64, 256):
    hnsw.hnsw.efSearch = ef; bench(hnsw, f"HNSW M=32 ef={ef}")

quant = faiss.IndexFlatIP(d)
ivf = faiss.IndexIVFFlat(quant, d, 1024, faiss.METRIC_INNER_PRODUCT)
ivf.train(xb); ivf.add(xb)
for p in (1, 8, 64):
    ivf.nprobe = p; bench(ivf, f"IVF nlist=1024 nprobe={p}")
```

Plot **recall vs latency** for each setting. Also record memory and build time. (Random vectors are harder than real embeddings; repeat with your real chunk vectors.)

### Step 3: Sparse retrieval: BM25 and SPLADE

**BM25** scores documents by term frequency (with saturation), inverse document frequency (rare terms matter more) and length normalization. It's exact, explainable and great for IDs, codes, names and jargon.

```python
from rank_bm25 import BM25Okapi
import re
tok = lambda s: re.findall(r"[a-z0-9\-]+", s.lower())
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
scores = bm25.get_scores(tok("HX-4410 torque"))
```

**SPLADE** is a *learned* sparse model. A transformer outputs weights over the vocabulary, including expansion terms not in the text. It sits between BM25 and dense retrieval. Supported natively in Qdrant, Elasticsearch/OpenSearch and Milvus.

Try learned sparse locally with **bge-m3**, which returns dense, sparse and multi-vector outputs in one pass:

```python
from FlagEmbedding import BGEM3FlagModel           # uv add FlagEmbedding
m3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)   # runs on your GPU
out = m3.encode(["What torque does part HX-4410 need?"],
                return_dense=True, return_sparse=True, return_colbert_vecs=True)
print(out["lexical_weights"][0])   # token → weight: a learned, inspectable sparse vector
```

### Step 4: Hybrid search and score fusion

Run dense and sparse retrieval in parallel, then merge the ranked lists.

**Reciprocal Rank Fusion (RRF)**: uses ranks only, so incompatible score scales (cosine 0–1 vs BM25 0–30) don't matter:

```
RRF(d) = Σ over retrievers  1 / (k + rank_i(d)),   k ≈ 60
```

```python
def rrf(result_lists, k=60):
    scores = {}
    for results in result_lists:                 # each is a list of chunk_ids in rank order
        for rank, cid in enumerate(results, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**Weighted convex combination**: `score = α·norm(dense) + (1−α)·norm(sparse)`. It needs score normalization (min-max or z-score) and tuning α. It can beat RRF when tuned, but it's more fragile.

#### Lab 3b: Build `rag/search.py`

```python
class HybridRetriever:
    def __init__(self, dense_store, bm25_index, k_candidates=50): ...
    def search(self, query: str, k: int = 10, filters: dict | None = None) -> list[dict]:
        # 1) dense top-50   2) BM25 top-50   3) RRF   4) return top-k with scores
        ...
```

Run the golden set: dense only vs BM25 only vs hybrid RRF vs weighted (α = 0.3/0.5/0.7). Break results down by question `type`, and watch the `identifier` questions jump.

### Step 5: Metadata filtering

Filters: tenant, user groups, date range, document type, language.

| | Pre-filtering (filter during ANN search) | Post-filtering (search, then filter) |
|---|---|---|
| Result count | Always returns k valid results | Can return fewer than k, or **zero**, with a strict filter |
| Speed | Engine-dependent; very selective filters can slow graph traversal | Fast search, wasted candidates |
| Security | ✅ Required for access control | ❌ Never for security |

Modern engines (Qdrant, Weaviate, OpenSearch efficient filtering, Milvus) do filtered HNSW traversal. pgvector with `WHERE` clauses can fall back to scans or lose recall; learn `hnsw.iterative_scan` and partial indexes.

### Step 6: Vector database landscape

Learn to choose with five factors: **scale, latency and QPS, hybrid needs, filtering and multi-tenancy, and the existing stack plus ops capacity.**

| Store | Type | Choose when |
|-------|------|-------------|
| FAISS | Library | Learning, benchmarks, in-process search |
| Chroma | Embedded DB | Prototypes, notebooks |
| **Qdrant** | Purpose-built, OSS | Strong filtering, hybrid (sparse + dense), multi-tenancy, self-host |
| Weaviate | Purpose-built, OSS | Built-in hybrid, modules |
| Milvus | Purpose-built, OSS | Billion scale, GPU indexes |
| **pgvector** | Postgres extension | Already on Postgres; vectors beside app data; SQL joins |
| OpenSearch / Elasticsearch | Search engine + vectors | Mature BM25 + k-NN hybrid, aggregations |
| Pinecone | Managed SaaS | Zero ops; serverless |
| S3 Vectors | Object storage + vectors | Huge, rarely queried, cost-sensitive; ~100 ms+ latency |

#### Lab 3c: Run Qdrant and pgvector in Docker

```bash
docker run -p 6333:6333 qdrant/qdrant
docker run -e POSTGRES_PASSWORD=dev -p 5432:5432 pgvector/pgvector:pg17
```

Load your chunks into both. In Qdrant, use **named vectors** (dense + sparse) and its native hybrid query. In pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (chunk_id text PRIMARY KEY, text text, page int,
                     tenant_id text, allowed_groups text[], embedding vector(1024));  -- Qwen3-Embedding-0.6B
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
SET hnsw.ef_search = 64;
SELECT chunk_id, page, 1 - (embedding <=> $1) AS similarity
FROM chunks WHERE tenant_id = 'acme' AND allowed_groups && ARRAY['finance']
ORDER BY embedding <=> $1 LIMIT 20;
```

Implement the same `VectorStore` interface for both, so the rest of the pipeline doesn't care which one runs.

### Step 7 (optional, advanced): Embedding fine-tuning

When off-the-shelf embeddings plateau on a niche domain (legal, medical, internal jargon):
- Generate (question, positive chunk) pairs from your corpus with the local **Qwen3-4B** helper (free, so you can generate thousands); mine hard negatives with BM25.
- Fine-tune with `sentence-transformers` using `MultipleNegativesRankingLoss`.
- Evaluate on a **held-out** golden split. Gains of 5–15% recall on niche domains are common, but measure yours.

---

## Milestone

- ANN benchmark notebook: recall vs latency vs memory plots for Flat, HNSW (3 settings), IVF (3 settings), IVF-PQ or binary + rescoring
- `rag/search.py` with dense, BM25, and hybrid (RRF + weighted) retrieval, plus metadata filters
- Your pipeline running on Qdrant **and** pgvector behind the same interface
- Results table by question type (identifier questions should improve with hybrid)

## Exit checklist

- [ ] I can explain HNSW with a drawing and name `M`, `ef_construction`, `ef_search`.
- [ ] I can explain IVF and PQ and when to use them.
- [ ] I can explain why vector search misses product codes and how BM25 fixes it.
- [ ] I can write RRF from memory and explain why it needs no normalization.
- [ ] I can explain why security filters must be pre-filters.
- [ ] I can pick a vector DB for a scenario and justify it with the five factors.

## Common pitfalls

- Benchmarking ANN with no brute-force ground truth.
- Tuning `ef_search` once and forgetting it: it often matters more than `M`.
- Using post-filtering for tenant isolation.
- Quantizing without re-checking recall.
- Picking a vector DB because it's popular rather than because it fits the requirements.

## Interview questions

1. Explain HNSW and its tuning knobs.
2. Pre-filtering vs post-filtering?
3. What is hybrid search and how do you combine the scores?
4. Why does vector search miss product codes?
5. Which vector store would you choose for (a) a Postgres shop at 2M chunks, (b) 1B vectors, rarely queried, (c) heavy keyword queries with filters?

## Resources

- Malkov & Yashunin: HNSW paper; Pinecone Learn: *Faiss* and *HNSW* series
- Jégou et al.: *Product Quantization for Nearest Neighbor Search*
- Formal et al.: *SPLADE* papers
- Cormack et al., 2009: *Reciprocal Rank Fusion*
- Qdrant hybrid-search docs; pgvector README; ann-benchmarks.com
- Guide: Part 1 §6; Interview Q11–Q16

## Next

→ [Phase 04: Query Transformation & Pre-Retrieval Optimization](../04-query-transformation/README.md)
