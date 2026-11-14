# compare_stores.py: Lab 1d
# Load the same chunks into NumPy, FAISS and Chroma, run the same questions,
# and check they return the same results. Only storage and search change.
#
# Run from rag-lab/ (after naive_rag.py built data/indexes/phase01/numpy/naive_800/):
#     uv run python labs/phase01_naive_rag/lab1d_vector_stores/compare_stores.py

import json
import time
from pathlib import Path

import numpy as np
from rag.embed import embed_query
from rag.vector_store import ChromaStore, FaissStore, NumpyStore

TOP_K = 5
CHUNKS_FILE = Path("data/indexes/phase01/numpy/naive_800/chunks.jsonl")
VECTORS_FILE = Path("data/indexes/phase01/numpy/naive_800/vectors.npy")
GOLDEN_FILE = Path("eval/golden.jsonl")
CHROMA_DIR = "data/indexes/phase01/chroma"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_stores(chunks, vectors):
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [{"source": c["source"], "page": c["page"], "text": c["text"]} for c in chunks]

    numpy_store = NumpyStore()
    numpy_store.add(ids, vectors, metadatas)

    faiss_store = FaissStore(dim=vectors.shape[1])
    faiss_store.add(ids, vectors, metadatas)

    chroma_store = ChromaStore(path=CHROMA_DIR, collection="naive_800")
    if chroma_store.count() == len(ids):
        print(f"Chroma: loaded {chroma_store.count()} chunks from disk ({CHROMA_DIR})")
    else:
        chroma_store.add(ids, vectors, metadatas)
        print(f"Chroma: wrote {chroma_store.count()} chunks to disk ({CHROMA_DIR})")

    return {"numpy": numpy_store, "faiss": faiss_store, "chroma": chroma_store}


def compare_on_golden(stores, questions):
    """Run each question through every store; count how often they agree with NumPy."""
    agree = {name: 0 for name in stores}
    seconds = {name: 0.0 for name in stores}

    for question in questions:
        query_vector = embed_query(question)          # embed once, search everywhere
        results = {}
        for name, store in stores.items():
            start = time.perf_counter()
            results[name] = store.search(query_vector, TOP_K)
            seconds[name] += time.perf_counter() - start

        reference_ids = [id for id, _ in results["numpy"]]
        for name in stores:
            if [id for id, _ in results[name]] == reference_ids:
                agree[name] += 1

    print(f"\nSame top-{TOP_K} as NumPy, and search time (embedding not included):")
    for name in stores:
        avg_ms = 1000 * seconds[name] / len(questions)
        print(f"  {name:7} {agree[name]:2}/{len(questions)} questions   {avg_ms:6.2f} ms per search")


def show_one_question(stores, question, where=None):
    print(f"\nQuestion: {question}")
    if where:
        print(f"Filter:   {where}")
    query_vector = embed_query(question)
    for name, store in stores.items():
        print(f"  {name}:")
        for id, score in store.search(query_vector, 3, where=where):
            print(f"    {score:.3f}  {id}")


def main():
    chunks = read_jsonl(CHUNKS_FILE)
    vectors = np.load(VECTORS_FILE)
    stores = build_stores(chunks, vectors)

    questions = [row["question"] for row in read_jsonl(GOLDEN_FILE)]
    compare_on_golden(stores, questions)

    show_one_question(stores, "How does the model handle the order of words?")
    show_one_question(stores, "How does the model handle the order of words?",
                      where={"source": "2005.11401v4.pdf"})


if __name__ == "__main__":
    main()
