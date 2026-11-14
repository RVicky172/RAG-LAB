# eval/

How well does retrieval work, and did a change make it better? One folder per phase. Shared pieces stay at the top.

```
eval/
├── README.md              ← this file
├── golden.jsonl           ← the golden questions (shared by every phase; grows over time)
├── eval_core.py           ← shared scoring: relevance rule, hit@5, MRR, report
├── results.md             ← summary: best config per phase, links to details
│
├── phase01_naive/         ← Phase 01: naive RAG baseline (one fixed index)
│   ├── eval_numpy.py      ┐
│   ├── eval_faiss.py      ├ same golden set, three vector stores
│   ├── eval_chroma.py     ┘
│   ├── results.md         ← Phase 01 numbers
│   └── failures.md        ← the 5 failure analyses (F1–F5), status updated as later phases fix them
│
└── phase02_chunking/      ← Phase 02: parser + chunker ablation (one index per chunk file)
    ├── common.py          ← load chunks, embed once (cached), score, record
    ├── eval_numpy.py      ┐
    ├── eval_faiss.py      ├ same golden set, three vector stores
    ├── eval_chroma.py     ┘
    ├── results.md         ← ablation table, by question type, decision
    └── runs.jsonl         ← raw log: one line per (store, config) run, append-only
```

## Run (from `rag-lab/`, LM Studio server running)

```bash
# Phase 01 baseline
uv run python -m eval.phase01_naive.eval_numpy

# Phase 02: every chunk file in data/chunks/, or name some
uv run python -m eval.phase02_chunking.eval_numpy
uv run python -m eval.phase02_chunking.eval_numpy docling-structure-512-64 --verbose
```

`numpy`, `faiss` and `chroma` give the same numbers at this corpus size (all cosine similarity; HNSW finds the exact neighbours on ~200 vectors). Use NumPy for quick runs.

## Where the indexes live

| Phase | NumPy | FAISS | Chroma |
|-------|-------|-------|--------|
| 01 | `data/indexes/phase01/numpy/naive_800/` | `data/indexes/phase01/faiss/naive_800/` | `data/indexes/phase01/chroma/` (collection `naive_800`) |
| 02 | in memory | `data/indexes/phase02/faiss/<config>/` | `data/indexes/phase02/chroma/` (one collection per config) |

Phase 02 embeddings are cached in `data/cache/embeddings/`, so rebuilding any index costs no embedding calls.

## Adding a phase
Create `phaseNN_<topic>/` with its own scripts, `results.md` and raw run log; reuse `eval_core.py`; add one row to the top-level `results.md`.
