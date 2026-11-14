# data/

Everything here is git-ignored except this file. Folders are in pipeline order; every path is defined once in [`src/rag/paths.py`](../src/rag/paths.py).

| Folder | Produced by | Contents | Safe to delete? |
|--------|-------------|----------|-----------------|
| `raw/` | you | source PDFs (the corpus) | **No**: it's the input |
| `parsed/<parser>/` | `uv run python -m rag.parse data/raw [--parser pypdf]` | `<doc_id>.json`: typed elements with page + section path | Yes (fast to rebuild) |
| `figures/<doc_id>/` | `rag.parse` (docling) | `pNNN-figN.png` crops for vision captions | Yes |
| `chunks/` | `uv run python -m rag.chunk <chunker>` | `<parser>-<chunker>.jsonl`, one chunk per line | Yes |
| `cache/embeddings/` | eval scripts | chunk vectors, keyed by chunk file + model + content hash | Yes, but re-embedding costs time |
| `cache/contexts.jsonl` | `rag.chunk contextual` | LLM-written chunk contexts, keyed by model + chunk hash | Avoid: ~16 min of LLM time |
| `indexes/<phase>/<store>/` | lab and eval scripts | NumPy / FAISS / Chroma indexes per phase | Yes |
| `lab_outputs/<phase>/<lab>/` | `labs/...` scripts | one-off experiment output (e.g. lab2a `inspect/`, `shootout/`) | Yes |

Rebuild everything from `raw/` (from `rag-lab/`):

```bash
uv run python -m rag.parse data/raw --force
uv run python -m rag.parse data/raw --parser pypdf --force
uv run python -m rag.chunk recursive --parser pypdf
uv run python -m rag.chunk recursive
uv run python -m rag.chunk structure
uv run python -m rag.chunk semantic
uv run python -m rag.chunk contextual --size 400
uv run python -m eval.phase02_chunking.eval_numpy
```
