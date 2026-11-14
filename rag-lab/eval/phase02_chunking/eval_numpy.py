# eval/phase02_chunking/eval_numpy.py
# Phase 02 chunker ablation on the NumPy store: brute-force exact search,
# rebuilt in memory for every chunk file (nothing saved).
#
# Run from rag-lab/ (LM Studio server running, embedding model available):
#     uv run python -m eval.phase02_chunking.eval_numpy                       # every chunk file
#     uv run python -m eval.phase02_chunking.eval_numpy docling-structure-512-64 --verbose

from eval.phase02_chunking.common import run, store_metadata
from rag.vector_store import NumpyStore


def build_numpy_store(config, chunks, vectors):
    store = NumpyStore()
    store.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        vectors=vectors,
        metadatas=[store_metadata(chunk) for chunk in chunks],
    )
    return store


if __name__ == "__main__":
    run("NumPy (exact)", build_numpy_store)
