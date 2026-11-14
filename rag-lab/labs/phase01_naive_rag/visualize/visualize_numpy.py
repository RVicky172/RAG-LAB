# visualize_numpy.py
# Chart of the chunks stored as vectors.npy + chunks.jsonl (the lab1c NumPy version).
#
# Needs: data/indexes/phase01/numpy/naive_800/  (built by labs/phase01_naive_rag/lab1c_naive_rag/naive_rag.py)
# Run from rag-lab/ (LM Studio must be running to embed the question):
#     uv run python labs/phase01_naive_rag/visualize/visualize_numpy.py        # 3D
#     uv run python labs/phase01_naive_rag/visualize/visualize_numpy.py 2d     # 2D

import json
from pathlib import Path

import numpy as np
from rag.vector_store import NumpyStore
from visualize_core import dimensions_from_command_line, plot_chunks

NUMPY_FOLDER = Path("data/indexes/phase01/numpy/naive_800")


def read_numpy_store():
    """With NumPy, the vectors and the chunk text live in two separate files."""
    chunk_vectors = np.load(NUMPY_FOLDER / "vectors.npy")
    chunks = [json.loads(line) for line in
              (NUMPY_FOLDER / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(chunks) != len(chunk_vectors):
        raise SystemExit("chunks.jsonl and vectors.npy are out of sync: re-run naive_rag.py")

    chunk_ids = [chunk["id"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "page": chunk["page"], "text": chunk["text"]}
                 for chunk in chunks]

    store = NumpyStore()
    store.add(chunk_ids, chunk_vectors, metadatas)
    return store, chunk_ids, chunk_vectors, metadatas


if __name__ == "__main__":
    store, chunk_ids, chunk_vectors, metadatas = read_numpy_store()
    plot_chunks(store, chunk_ids, chunk_vectors, metadatas,
                store_name="NumPy", dimensions=dimensions_from_command_line())
