# eval/phase01_naive/eval_numpy.py  (Phase 01 baseline: naive 800-char chunks)
# Retrieval metrics (hit@5, MRR) using the NumPy store: brute-force search
# over vectors.npy, the same maths as lab1c/naive_rag.py.
#
# Needs: data/indexes/phase01/numpy/naive_800/  (built by labs/phase01_naive_rag/lab1c_naive_rag/naive_rag.py)
# Run from rag-lab/:
#     uv run python -m eval.phase01_naive.eval_numpy

from pathlib import Path

import numpy as np
from eval.eval_core import evaluate, read_jsonl, report
from rag.vector_store import NumpyStore

NUMPY_FOLDER = Path("data/indexes/phase01/numpy/naive_800")


def open_numpy_store():
    chunks = read_jsonl(NUMPY_FOLDER / "chunks.jsonl")
    chunk_vectors = np.load(NUMPY_FOLDER / "vectors.npy")
    if len(chunks) != len(chunk_vectors):
        raise SystemExit("chunks.jsonl and vectors.npy are out of sync: re-run naive_rag.py")

    store = NumpyStore()
    store.add(
        ids=[chunk["id"] for chunk in chunks],
        vectors=chunk_vectors,
        metadatas=[{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks],
    )
    print(f"NumPy: loaded {len(chunks)} chunks from {NUMPY_FOLDER}")
    return store


if __name__ == "__main__":
    store = open_numpy_store()
    report(evaluate(store), store_name="NumPy (exact search)")
