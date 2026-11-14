# eval/phase01_naive/eval_chroma.py  (Phase 01 baseline: naive 800-char chunks)
# Retrieval metrics (hit@5, MRR) using the Chroma store: HNSW, approximate
# search, loaded from disk.
#
# Needs: data/indexes/phase01/chroma/ with collection "naive_800"  (built by labs/phase01_naive_rag/lab1d_vector_stores/rag_with_chroma.py)
# Run from rag-lab/:
#     uv run python -m eval.phase01_naive.eval_chroma

from eval.eval_core import evaluate, report
from rag.vector_store import ChromaStore

CHROMA_FOLDER = "data/indexes/phase01/chroma"
COLLECTION_NAME = "naive_800"


def open_chroma_store():
    store = ChromaStore(path=CHROMA_FOLDER, collection=COLLECTION_NAME)
    if store.count() == 0:
        raise SystemExit(f"Collection '{COLLECTION_NAME}' is empty: run labs/phase01_naive_rag/lab1d_vector_stores/rag_with_chroma.py first")

    print(f"Chroma: loaded {store.count()} chunks from {CHROMA_FOLDER} (collection '{COLLECTION_NAME}')")
    return store


if __name__ == "__main__":
    store = open_chroma_store()
    report(evaluate(store), store_name="Chroma (HNSW, approximate search)")
