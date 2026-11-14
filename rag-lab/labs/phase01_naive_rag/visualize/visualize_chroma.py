# visualize_chroma.py
# Chart of the chunks stored in the Chroma database.
#
# Needs: data/indexes/phase01/chroma/ with collection "naive_800"  (built by labs/phase01_naive_rag/lab1d_vector_stores/rag_with_chroma.py)
# Run from rag-lab/ (LM Studio must be running to embed the question):
#     uv run python labs/phase01_naive_rag/visualize/visualize_chroma.py        # 3D
#     uv run python labs/phase01_naive_rag/visualize/visualize_chroma.py 2d     # 2D

import numpy as np
from rag.vector_store import ChromaStore
from visualize_core import dimensions_from_command_line, plot_chunks

CHROMA_FOLDER = "data/indexes/phase01/chroma"
COLLECTION_NAME = "naive_800"


def read_chroma_store():
    store = ChromaStore(path=CHROMA_FOLDER, collection=COLLECTION_NAME)
    if store.count() == 0:
        raise SystemExit(f"Collection '{COLLECTION_NAME}' is empty: run labs/phase01_naive_rag/lab1d_vector_stores/rag_with_chroma.py first")

    # Chroma does NOT return the vectors unless we ask for them with include=.
    # Note: the order is not PDF order, but ids, vectors and metadata stay matched.
    stored_records = store.collection.get(include=["embeddings", "metadatas"])
    chunk_vectors = np.array(stored_records["embeddings"])
    return store, stored_records["ids"], chunk_vectors, stored_records["metadatas"]


if __name__ == "__main__":
    store, chunk_ids, chunk_vectors, metadatas = read_chroma_store()
    plot_chunks(store, chunk_ids, chunk_vectors, metadatas,
                store_name="Chroma", dimensions=dimensions_from_command_line())
