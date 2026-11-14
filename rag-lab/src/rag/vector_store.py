# rag/vector_store.py
# One interface, three storage backends. The RAG logic only ever calls:
#     store.add(ids, vectors, metadatas)
#     store.search(query_vector, k, where=None) -> [(id, score), ...]
#
# Every backend returns the SAME kind of score: cosine similarity (higher = better),
# so results can be compared directly.

from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int,
               where: dict | None = None) -> list[tuple[str, float]]: ...

    @abstractmethod
    def get_metadata(self, id: str) -> dict: ...


def matches(metadata: dict, where: dict | None) -> bool:
    """Simple equality filter: {"source": "a.pdf"} keeps chunks whose source is a.pdf."""
    if not where:
        return True
    return all(metadata.get(key) == value for key, value in where.items())


# ---------- 1. NumPy: what you had in naive_rag.py ----------

class NumpyStore(VectorStore):
    def __init__(self):
        self.ids: list[str] = []
        self.metadatas: list[dict] = []
        self.vectors = np.empty((0, 0), dtype="float32")

    def add(self, ids, vectors, metadatas):
        self.ids += ids
        self.metadatas += metadatas
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])

    def search(self, query_vector, k, where=None):
        scores = self.vectors @ query_vector            # compare with every vector
        ranked = np.argsort(-scores)
        results = []
        for i in ranked:                                # filter, then keep the top k
            if matches(self.metadatas[i], where):
                results.append((self.ids[i], float(scores[i])))
            if len(results) == k:
                break
        return results

    def get_metadata(self, id):
        return self.metadatas[self.ids.index(id)]


# ---------- 2. FAISS: a fast search library, nothing more ----------

class FaissStore(VectorStore):
    """
    FAISS only stores vectors and only knows them by position (0, 1, 2 ...).
    So WE keep the ids and metadata in Python lists, in the same order.
    IndexFlatIP = exact search by inner product (= cosine for normalized vectors).
    """

    def __init__(self, dim: int):
        import faiss
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, ids, vectors, metadatas):
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.ids += ids
        self.metadatas += metadatas

    def search(self, query_vector, k, where=None):
        # FAISS has no metadata filter. With a filter we fetch everything and
        # filter afterwards. Fine at this size; at scale you'd use a FAISS
        # IDSelector or a database that filters for you (Chroma, Qdrant).
        fetch = self.index.ntotal if where else k
        scores, positions = self.index.search(query_vector.reshape(1, -1).astype("float32"), fetch)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:                               # FAISS pads with -1 when it runs out
                continue
            if matches(self.metadatas[pos], where):
                results.append((self.ids[pos], float(score)))
            if len(results) == k:
                break
        return results

    def get_metadata(self, id):
        return self.metadatas[self.ids.index(id)]

    def count(self) -> int:
        return self.index.ntotal

    # FAISS can save only the index (the vectors). The ids and metadata are
    # OUR job, so we save them next to it in a second file, in the same order.

    def save(self, folder: str) -> None:
        import json
        from pathlib import Path
        import faiss

        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder_path / "index.faiss"))
        with open(folder_path / "records.jsonl", "w", encoding="utf-8") as records_file:
            for chunk_id, metadata in zip(self.ids, self.metadatas):
                records_file.write(json.dumps({"id": chunk_id, "metadata": metadata}) + "\n")

    @classmethod
    def load(cls, folder: str) -> "FaissStore":
        import json
        from pathlib import Path
        import faiss

        folder_path = Path(folder)
        index = faiss.read_index(str(folder_path / "index.faiss"))
        records = [json.loads(line) for line in
                   (folder_path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(records) != index.ntotal:
            raise ValueError(f"{folder}: index has {index.ntotal} vectors but "
                             f"records.jsonl has {len(records)} rows. Rebuild it.")

        store = cls(dim=index.d)
        store.index = index
        store.ids = [record["id"] for record in records]
        store.metadatas = [record["metadata"] for record in records]
        return store

    @staticmethod
    def exists(folder: str) -> bool:
        from pathlib import Path
        return (Path(folder) / "index.faiss").exists() and (Path(folder) / "records.jsonl").exists()


# ---------- 3. Chroma: a real database, saved to disk ----------

class ChromaStore(VectorStore):
    """
    Chroma stores the vector, the id and the metadata TOGETHER, saves them to
    disk, and can filter by metadata itself. It searches with HNSW, an
    approximate index (fast, very rarely misses a true neighbour).
    """

    def __init__(self, path: str, collection: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, ids, vectors, metadatas, batch_size: int = 1000):
        # upsert = insert, or overwrite if the id already exists, so re-running is safe
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(ids=ids[start:end],
                                   embeddings=vectors[start:end].tolist(),
                                   metadatas=metadatas[start:end])

    def search(self, query_vector, k, where=None):
        result = self.collection.query(query_embeddings=[query_vector.tolist()],
                                       n_results=k,
                                       where=self._chroma_filter(where))
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma returns cosine DISTANCE (0 = identical). Convert to similarity.
        return [(id, 1 - distance) for id, distance in zip(ids, distances)]

    def get_metadata(self, id):
        return self.collection.get(ids=[id])["metadatas"][0]

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _chroma_filter(where):
        """Chroma wants {"a": 1} for one condition and {"$and": [...]} for several."""
        if not where:
            return None
        if len(where) == 1:
            return where
        return {"$and": [{key: value} for key, value in where.items()]}
