# visualize_faiss.py
# Chart of the chunks stored in the FAISS index.
#
# Needs: data/indexes/phase01/faiss/naive_800/  (built by labs/phase01_naive_rag/lab1d_vector_stores/rag_with_faiss.py)
# Run from rag-lab/ (LM Studio must be running to embed the question):
#     uv run python labs/phase01_naive_rag/visualize/visualize_faiss.py        # 3D
#     uv run python labs/phase01_naive_rag/visualize/visualize_faiss.py 2d     # 2D

from rag.vector_store import FaissStore
from visualize_core import dimensions_from_command_line, plot_chunks

FAISS_FOLDER = "data/indexes/phase01/faiss/naive_800"


def read_faiss_store():
    if not FaissStore.exists(FAISS_FOLDER):
        raise SystemExit(f"No FAISS index in {FAISS_FOLDER}: run labs/phase01_naive_rag/lab1d_vector_stores/rag_with_faiss.py first")

    store = FaissStore.load(FAISS_FOLDER)

    # FAISS can give the vectors back with reconstruct_n(start, count).
    # They come out by position 0, 1, 2 ... which matches the order of
    # store.ids and store.metadatas (the records.jsonl file).
    chunk_vectors = store.index.reconstruct_n(0, store.count())
    return store, store.ids, chunk_vectors, store.metadatas


if __name__ == "__main__":
    store, chunk_ids, chunk_vectors, metadatas = read_faiss_store()
    plot_chunks(store, chunk_ids, chunk_vectors, metadatas,
                store_name="FAISS", dimensions=dimensions_from_command_line())
