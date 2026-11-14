# Results summary

The best configuration of each phase. Details, per-question-type tables and decisions live in each phase's own `results.md`.

| Phase | Best config | Corpus | Golden | hit@5 | MRR | Details |
|-------|-------------|--------|--------|-------|-----|---------|
| 01 Naive RAG | pypdf, 800-char / 150 overlap, nomic-v1.5 | 2 PDFs | v2 (18 answerable) | 0.78 | 0.72 | [phase01_naive/results.md](phase01_naive/results.md) |
| 02 Ingestion & chunking | Docling + structure-aware 400/64 + section path + **vision captions**, nomic-v1.5 (contextual retrieval: inconsistent, not adopted; Qwen3-Embedding-4B ties, 0.6B worse) | 4 PDFs | **v4 (53 answerable, incl. 15 chart)** | 0.92 | 0.80 | [phase02_chunking/results.md](phase02_chunking/results.md) |

> Rows are only comparable when corpus **and** golden version match. Phase 02 history: golden v2 0.89 / 0.75; v3 (+21 questions on the two-column papers) 0.92 / 0.78; v4 (+15 chart questions) 0.91 / 0.76; + vision captions 0.92 / 0.80 (chart fig@5 0.53 → 0.93). Phase 02 doubled the corpus, so its own `pypdf-recursive` row (0.72 / 0.69) is the baseline to compare against, not the Phase 01 row.

## Open failures
Tracked in [phase01_naive/failures.md](phase01_naive/failures.md). After Phase 02: F2 (q012, diluted chunk) is **fixed**; paraphrased misses q013, q021 remain; new on golden v3: q023 (Contriever: "distractor documents" matches a whole appendix instead of p.3). Phases 03–05.
