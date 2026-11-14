# eval/phase02_chunking/eval_chroma.py
# Phase 02 chunker ablation on the Chroma store: HNSW (approximate search),
# one collection per chunk file in data/indexes/phase02/chroma/, rebuilt when the chunks change.
#
# Run from rag-lab/ (LM Studio server running, embedding model available):
#     uv run python -m eval.phase02_chunking.eval_chroma                      # every chunk file
#     uv run python -m eval.phase02_chunking.eval_chroma docling-structure-512-64 --verbose

from eval.phase02_chunking.common import run, store_metadata
from rag.vector_store import ChromaStore

CHROMA_FOLDER = "data/indexes/phase02/chroma"


def build_chroma_store(config, chunks, vectors):
    ids = [chunk["chunk_id"] for chunk in chunks]
    store = ChromaStore(path=CHROMA_FOLDER, collection=config)

    # Same ids already stored? Reuse. Otherwise start the collection from scratch,
    # so chunks from an older run can't linger and pollute the results.
    if sorted(store.collection.get(include=[])["ids"]) != sorted(ids):
        store.client.delete_collection(config)
        store = ChromaStore(path=CHROMA_FOLDER, collection=config)
        store.add(ids, vectors, [store_metadata(chunk) for chunk in chunks])
    return store


if __name__ == "__main__":
    run("Chroma (HNSW, approximate)", build_chroma_store)
