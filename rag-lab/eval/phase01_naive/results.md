# Phase 01 results: naive RAG baseline

**Setup:** 800-char chunks / 150 overlap (pypdf) · nomic-embed-text-v1.5 · top-5 · corpus: 2 PDFs (1706.03762, 2005.11401).
**Scripts:** `uv run python -m eval.phase01_naive.eval_numpy` (also `eval_faiss`, `eval_chroma`; all give the same numbers).
**Failure analysis:** [failures.md](failures.md)

| Config                         | hit@5 | MRR  | notes |
|--------------------------------|-------|------|-------|
| 01 naive: 800-char, nomic-v1.5 | 0.82  | 0.76 | golden v1 (20 q, 17 answerable). 3/17 missed: paraphrased 0.33 (q013, q016), multi-part 0.75 (q012) |
| 01 naive: 800-char, nomic-v1.5 | 0.78  | 0.72 | golden v2 (21 q, +q021). Same system; score dropped only because q021 was added and misses. 4/18 missed: paraphrased 0.25 (q013, q016, q021), multi-part 0.75 (q012) |

> Only compare rows that use the same golden version. From now on, compare against the **v2** row.


