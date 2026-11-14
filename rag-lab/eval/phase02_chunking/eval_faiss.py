# eval/phase02_chunking/eval_faiss.py
# Phase 02 chunker ablation on the FAISS store: IndexFlatIP (exact search),
# one index per chunk file saved to data/indexes/phase02/faiss/<config>/ and reused on re-runs.
#
# Run from rag-lab/ (LM Studio server running, embedding model available):
#     uv run python -m eval.phase02_chunking.eval_faiss                       # every chunk file
#     uv run python -m eval.phase02_chunking.eval_faiss docling-structure-512-64 --verbose

from eval.phase02_chunking.common import run, store_metadata
from rag.vector_store import FaissStore

FAISS_ROOT = "data/indexes/phase02/faiss"


def build_faiss_store(config, chunks, vectors):
    folder = f"{FAISS_ROOT}/{config}"
    ids = [chunk["chunk_id"] for chunk in chunks]

    # Reuse the saved index only if it holds exactly these chunks
    if FaissStore.exists(folder):
        store = FaissStore.load(folder)
        if store.ids == ids:
            return store

    store = FaissStore(dim=vectors.shape[1])
    store.add(ids, vectors, [store_metadata(chunk) for chunk in chunks])
    store.save(folder)
    return store


if __name__ == "__main__":
    run("FAISS (IndexFlatIP, exact)", build_faiss_store)
