# rag-lab

The code for the RAG learning plan (see the [roadmap](../README.md)). Run everything from this folder with `uv run`.

```
rag-lab/
├── pyproject.toml  uv.lock  .python-version  .env   ← environment (uv) and model settings
│
├── src/rag/                  ← THE LIBRARY: reusable pipeline code, grows phase by phase
│   ├── llm.py                   LLM wrapper: LM Studio now, Bedrock in Phase 09        (01)
│   ├── embed.py                 embedding wrapper + query/document prefixes            (01)
│   ├── vector_store.py          one interface: NumPy, FAISS, Chroma                    (01)
│   ├── parse.py                 PDF -> typed elements (Docling; pypdf baseline)        (02)
│   ├── chunk.py                 recursive, structure-aware, semantic chunkers          (02)
│   ├── enrich.py                contextual retrieval                                   (02)
│   └── caption.py               vision-LLM figure descriptions                         (02)
│
├── labs/                     ← LEARNING SCRIPTS: experiments you run once, per phase
│   ├── phase00_prerequisites/   hello_llm, token_count, estimate, vector, prompting
│   ├── phase01_naive_rag/
│   │   ├── lab1a_embeddings/    weak vs strong embedding model, truncation test
│   │   ├── lab1c_naive_rag/     naive_rag.py: RAG from scratch with NumPy
│   │   ├── lab1d_vector_stores/ same pipeline on FAISS and Chroma (findings: README.md)
│   │   └── visualize/           plot chunk embeddings in 2D/3D
│   └── phase02_ingestion/
│       └── lab2a_parsing/       inspect_pages, parser_shootout, defects.md, shootout.md
│
├── eval/                     ← MEASUREMENT: golden set + one folder per phase (see eval/README.md)
├── tests/                    ← unit tests for src/rag (uv run pytest)
│
└── data/                     ← git-ignored, in pipeline order (all paths defined in src/rag/paths.py)
    ├── raw/                     1. source PDFs (the corpus): the only thing you can't regenerate
    ├── parsed/<parser>/         2. rag.parse:  docling/<doc_id>.json, pypdf/<doc_id>.json
    ├── figures/<doc_id>/        2. rag.parse:  figure crops for vision captions
    ├── chunks/                  3. rag.chunk:  <parser>-<chunker>.jsonl
    ├── cache/                      expensive model outputs, keyed by content hash
    │   ├── embeddings/             chunk vectors (eval)
    │   ├── contexts.jsonl          contextual-retrieval sentences (rag.enrich)
    │   └── captions.jsonl          figure descriptions (rag.caption)
    ├── indexes/                 4. vector indexes
    │   ├── phase01/                numpy/, faiss/, chroma/ (naive_800)
    │   └── phase02/                faiss/<config>/, chroma/ (one collection per chunk file)
    └── lab_outputs/<phase>/<lab>/  one-off experiment outputs (e.g. lab2a inspect/, shootout/)
```

**The rules:**
- Code that the pipeline reuses goes in `src/rag/`; a one-off experiment goes in `labs/phaseNN_*/`; anything that produces a number for `results.md` goes in `eval/phaseNN_*/`.
- Never hard-code a data path: import it from `rag.paths`. Pipeline output goes in the numbered stage folders; lab output goes in `data/lab_outputs/`.
- Deleting anything except `data/raw/` is safe: re-run the stage that produces it. Keep `data/cache/` if you can: rebuilding it costs embedding and LLM time (contexts: ~16 min).

## Common commands

```bash
# Ingestion (Phase 02)
uv run python -m rag.parse data/raw                      # PDFs -> data/parsed/docling/
uv run python -m rag.chunk structure --size 512           # -> data/chunks/
uv run python -m rag.chunk contextual --size 400          # needs LM Studio (helper model)
uv run python -m rag.caption                              # needs the vision model (see CLAUDE.md)
uv run python -m rag.chunk structure --size 400 --captions # Phase 02 final config

# Evaluation
uv run python -m eval.phase02_chunking.eval_numpy         # every chunk file vs golden set
uv run pytest                                             # unit tests

# Labs (examples)
uv run python labs/phase00_prerequisites/hello_llm.py
uv run python labs/phase02_ingestion/lab2a_parsing/inspect_pages.py data/raw/2307.03172.pdf
```

Each session: start LM Studio (`lms server start`) and load only the models the step needs. Idle large models fill the 16 GB of VRAM and slow everything else down.
