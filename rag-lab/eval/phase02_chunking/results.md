# Phase 02 results: parser + chunker ablation


**Setup:** golden **v2** (21 q, 18 answerable) · top-5 · embeddings nomic-embed-text-v1.5 · corpus **4 PDFs, 60 pages** (added 2307.03172 *Lost in the Middle* and 2309.15217 *RAGAS*, both two-column).
**Scripts:** `uv run python -m eval.phase02_chunking.eval_numpy` (also `eval_faiss`, `eval_chroma`; shared code in [common.py](common.py); every run is appended to [runs.jsonl](runs.jsonl)).
**Relevance:** a chunk counts if it's from the right PDF and **any** of its pages is a golden page (chunks can now span pages).

> Not comparable with the [Phase 01 results](../phase01_naive/results.md): the corpus doubled (more distractors) and the chunk sizes changed. The `pypdf-recursive` row is the new baseline.

| Config | Parser | #chunks | avg tok | hit@5 | MRR | Missed |
|--------|--------|---------|---------|-------|-----|--------|
| 02 baseline: recursive 512/64 | pypdf | 117 | 487 | 0.72 | 0.69 | q011, q012, q013, q016, q021 |
| 02a recursive 512/64 | docling | 117 | 432 | 0.83 | 0.64 | q013, q016, q021 |
| 02c semantic p95, cap 512 | docling | 196 | 251 | 0.83 | 0.66 | q013, q016, q021 |
| 02b structure-aware + section path, 512/64 | docling | 202 | 262 | 0.83 | 0.74 | q013, q016, q021 |
| 02d structure-aware + contextual (Qwen3-4B) | docling | 202 | 332* | 0.83 | **0.77** | q013, q016, q021 |

\* 43 of 202 contextual chunks exceed 512 tokens (context adds ~70 tokens). Re-run with `--size 400` for a size-fair comparison.

**Same numbers on all three stores** (NumPy exact, FAISS IndexFlatIP exact, Chroma HNSW approximate). Expected: at ~200 vectors HNSW finds the true neighbours. The store choice matters for speed and features at scale (Phase 03), not for quality here.

### hit@5 / MRR by question type

| Config | fact (3) | identifier (3) | multi-part (4) | paraphrased (4) | table (4) |
|--------|----------|----------------|----------------|-----------------|-----------|
| pypdf recursive | 1.00 / 0.78 | 0.67 / 0.67 | 0.75 / 0.75 | 0.25 / 0.25 | 1.00 / 1.00 |
| docling recursive | 1.00 / 0.67 | 1.00 / 0.75 | 1.00 / 0.71 | 0.25 / 0.25 | 1.00 / 0.88 |
| docling semantic | 1.00 / 0.67 | 1.00 / 0.78 | 1.00 / 0.80 | 0.25 / 0.25 | 1.00 / 0.83 |
| docling structure | 1.00 / 0.78 | 1.00 / 1.00 | 1.00 / 0.88 | 0.25 / 0.25 | 1.00 / 0.88 |
| docling structure + contextual | 1.00 / 0.78 | 1.00 / 1.00 | 1.00 / 0.88 | 0.25 / 0.25 | 1.00 / 1.00 |

### What the numbers say

1. **The parser matters most (hit@5 0.72 → 0.83).** Docling fixed q011 (BART-large params, rank miss → rank 1 with structure) and **q012, failure F2 "diluted chunk"**: the Wikipedia-dump answer now sits in a focused chunk at rank 1.
2. **Chunking changes the ranking, not the hits (MRR 0.64 → 0.74 → 0.77).** Same 15/18 found by every Docling config; structure-aware puts the right chunk higher. Identifier questions go 0.75 → 1.00 MRR: the section path in the chunk ("RAG > 2.2 Generator") helps short lookups.
3. **Semantic ≈ recursive (0.66 vs 0.64).** On well-structured papers it adds embedding cost for no gain, as expected; it also crosses section boundaries.
4. **Contextual retrieval: small gain (0.74 → 0.77, table MRR 0.88 → 1.00)** for 16 minutes of LLM time on 60 pages. Promising but within noise on 18 questions.
5. **The 3 remaining misses are all paraphrased questions** (q013, q016, q021, same as Phase 01 F1/F3/F4). No chunker fixes word mismatch ("frozen" vs "fixed"); that's a query-side problem: hybrid search (Phase 03) and query rewriting / HyDE (Phase 04).

### Caveats
- **18 answerable questions:** one question = 0.056 hit@5. Differences under ~0.05 MRR are noise.
- **The golden set only covers the 2 original papers.** The two new two-column papers (where Docling helped most visually) and their tables/figures aren't tested yet.

### Decision
Use **Docling + structure-aware chunking (512 tokens, 64 overlap) with the section path prepended** as the default. It fixed the parsing failures, gives the best MRR without LLM cost, and keeps tables whole with their header. Keep **contextual retrieval** as the leading candidate: re-test it (at `--size 400`) once the golden set has 40 questions covering the two-column papers and their tables/figures. Drop semantic chunking for this corpus.
