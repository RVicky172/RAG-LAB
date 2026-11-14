# eval/phase01_naive/eval_faiss.py  (Phase 01 baseline: naive 800-char chunks)
# Retrieval metrics (hit@5, MRR) using the FAISS store: IndexFlatIP, exact
# search, loaded from disk.
#
# Needs: data/indexes/phase01/faiss/naive_800/  (built by labs/phase01_naive_rag/lab1d_vector_stores/rag_with_faiss.py)
# Run from rag-lab/:
#     uv run python -m eval.phase01_naive.eval_faiss

from eval.eval_core import evaluate, report
from rag.vector_store import FaissStore

FAISS_FOLDER = "data/indexes/phase01/faiss/naive_800"


def open_faiss_store():
    if not FaissStore.exists(FAISS_FOLDER):
        raise SystemExit(f"No FAISS index in {FAISS_FOLDER}: run labs/phase01_naive_rag/lab1d_vector_stores/rag_with_faiss.py first")

    store = FaissStore.load(FAISS_FOLDER)
    print(f"FAISS: loaded {store.count()} chunks from {FAISS_FOLDER}")
    return store


if __name__ == "__main__":
    store = open_faiss_store()
    report(evaluate(store), store_name="FAISS (IndexFlatIP, exact search)")
