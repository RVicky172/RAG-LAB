# Lab 1d: vector stores (NumPy vs FAISS vs Chroma)

**Question:** does the place we store and search vectors change which chunks we get back?
**Short answer:** not at our size. All three return the **same top-5 with the same scores**. They differ in speed, features and how much work they leave to you.

## What a vector store does

After embedding, every chunk is a list of numbers (a vector). To answer a question we embed the question too, then find the chunks whose vectors point in the most similar direction (**cosine similarity**, higher = more similar). A vector store keeps those vectors and does that search.

| Store | What it is | How it searches |
|-------|------------|-----------------|
| **NumPy** | a plain array in memory | **exact**: compares the question with *every* vector, then sorts |
| **FAISS** (`IndexFlatIP`) | a fast search *library* from Meta | **exact**: the same maths, optimised C++ |
| **Chroma** | a vector *database*, saved to disk | **approximate** (HNSW): walks a graph of nearest neighbours instead of checking every vector |

All three sit behind one interface in [`src/rag/vector_store.py`](../../../src/rag/vector_store.py): `add(ids, vectors, metadatas)` and `search(query_vector, k, where=None)`. Every store returns the same kind of score (cosine similarity), so results can be compared directly. Only the store changes; that is what makes this a fair test.

## How to run it

From `rag-lab/` (LM Studio running with the embedding model; needs the Phase 01 index from `naive_rag.py`):

```bash
uv run python labs/phase01_naive_rag/lab1d_vector_stores/compare_stores.py
uv run python labs/phase01_naive_rag/lab1d_vector_stores/rag_with_faiss.py    # full RAG on FAISS
uv run python labs/phase01_naive_rag/lab1d_vector_stores/rag_with_chroma.py   # full RAG on Chroma
```

## Results

Corpus: 183 chunks (Phase 01 naive 800-char chunks), nomic-embed-v1.5 (768 dims), all 57 golden questions, top-5. Measured 2026-11-14.

| Store | Same top-5 as NumPy | Time per search (embedding not included) |
|-------|--------------------|-----------------------------------------|
| NumPy | (reference) | **0.06 ms** |
| FAISS | 57 / 57 | 0.08 ms |
| Chroma | 57 / 57 | 1.37 ms |

Example, "How does the model handle the order of words?": all three return the same chunks with the same scores (0.738, 0.735, 0.730), with and without a metadata filter (`source = 2005.11401v4.pdf`).

The eval scripts confirm it at the metric level: every Phase 01 and Phase 02 eval (`eval_numpy`, `eval_faiss`, `eval_chroma`) gave **identical hit@5 and MRR** on all three stores.

## Findings

1. **Same quality at this size.** NumPy and FAISS-Flat are both exact, so they *must* agree. Chroma's HNSW is approximate, but with only ~200 vectors its graph search still finds the true nearest neighbours every time. Approximate search only starts to miss with very large collections or settings tuned for speed.
2. **The simplest store is the fastest here.** Chroma is ~20x slower per search, not because its search is worse, but because it is a full database (storage on disk, query layer, filtering). All three take under 2 ms; embedding the question takes far longer, so search speed doesn't matter yet.
3. **They differ in features, not answers.**
   - **NumPy:** no index, nothing saved, you write everything. Fine up to tens of thousands of vectors.
   - **FAISS:** only stores vectors. The ids and metadata are *our* job: we save them in a second file, in the same order. **No metadata filter**: our wrapper fetches all results and filters afterwards (OK at this size; at scale use an `IDSelector` or a database).
   - **Chroma:** a real database. Saves to disk, stores metadata, filters with `where`, and `upsert` makes re-runs safe. It returns cosine *distance* (0 = identical), which our wrapper converts to similarity.
4. **One interface made the test fair.** Because all three share `add()` / `search()` and the same score, identical numbers prove the store is not a variable. That's why Phase 02 ran its chunking evals on NumPy only.

## Which one when

| Situation | Pick |
|-----------|------|
| Learning, experiments, up to ~50K vectors | **NumPy**: simplest, exact, fastest at this size |
| Millions of vectors, speed matters, you manage metadata yourself | **FAISS** (approximate indexes: HNSW, IVF, PQ) |
| You need persistence, metadata filters, updates and deletes | **A vector database**: Chroma locally; Qdrant, pgvector, OpenSearch in production |

## Where it starts to matter (Phase 03)

With millions of vectors, checking every one is too slow, so you use approximate indexes. Then **speed, memory and recall** trade off: recall@k = how often the approximate top-k matches the exact top-k. Phase 03's ANN benchmark measures that, using NumPy exact search as the "true answer".
