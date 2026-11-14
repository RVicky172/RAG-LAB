"""rag/paths.py: every data location in one place.

Folders follow the pipeline order. Paths are relative to rag-lab/ (run
everything from there with `uv run`).

    data/
    ├── raw/                       1. source documents (the corpus)
    ├── parsed/<parser>/           2. rag.parse   -> <doc_id>.json (typed elements)
    ├── figures/<doc_id>/          2. rag.parse   -> figure crops for vision captions
    ├── chunks/                    3. rag.chunk   -> <parser>-<chunker>.jsonl
    ├── cache/                        expensive model outputs, keyed by content hash
    │   ├── embeddings/               chunk vectors (eval)
    │   └── contexts.jsonl            contextual-retrieval sentences (rag.enrich)
    ├── indexes/<phase>/<store>/   4. vector indexes built from chunks + embeddings
    └── lab_outputs/<phase>/<lab>/    one-off experiment outputs, not used by the pipeline
"""

from pathlib import Path

DATA = Path("data")

RAW = DATA / "raw"
PARSED = DATA / "parsed"
FIGURES = DATA / "figures"
CHUNKS = DATA / "chunks"

CACHE = DATA / "cache"
EMBEDDINGS_CACHE = CACHE / "embeddings"
CONTEXTS_CACHE = CACHE / "contexts.jsonl"

INDEXES = DATA / "indexes"
LAB_OUTPUTS = DATA / "lab_outputs"


def parsed_dir(parser: str) -> Path:
    """data/parsed/<parser>/: one folder per parser, same file names inside."""
    return PARSED / parser


def index_dir(phase: str, store: str) -> Path:
    """data/indexes/<phase>/<store>/, e.g. index_dir("phase02", "faiss")."""
    return INDEXES / phase / store


def lab_output_dir(phase: str, lab: str) -> Path:
    """data/lab_outputs/<phase>/<lab>/, e.g. lab_output_dir("phase02_ingestion", "lab2a_parsing")."""
    return LAB_OUTPUTS / phase / lab
