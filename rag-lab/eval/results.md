# Results summary

The best configuration of each phase. Details, per-question-type tables and decisions live in each phase's own `results.md`.

| Phase | Best config | Corpus | Golden | hit@5 | MRR | Details |
|-------|-------------|--------|--------|-------|-----|---------|
| 01 Naive RAG | pypdf, 800-char / 150 overlap, nomic-v1.5 | 2 PDFs | v2 (18 answerable) | 0.78 | 0.72 | [phase01_naive/results.md](phase01_naive/results.md) |
| 02 Ingestion & chunking | Docling + structure-aware 512/64 (+ contextual: MRR 0.77) | 4 PDFs | v2 (18 answerable) | 0.83 | 0.74 | [phase02_chunking/results.md](phase02_chunking/results.md) |

> Rows are only comparable when corpus **and** golden version match. Phase 02 doubled the corpus, so its own `pypdf-recursive` row (0.72 / 0.69) is the baseline to compare against, not the Phase 01 row.

## Open failures
Tracked in [phase01_naive/failures.md](phase01_naive/failures.md). After Phase 02: F2 (q012, diluted chunk) is **fixed**; the 3 paraphrased misses (q013, q016, q021) remain for Phases 03–04.
