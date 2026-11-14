# eval/eval_core.py
# Shared scoring code used by every eval script, in every phase:
#     phase01_naive/eval_*.py   (Phase 01: naive index)
#     phase02_chunking/eval_*.py (Phase 02: one index per chunk file)
#
# Each script only decides WHICH vector store to search. Everything here
# (what counts as relevant, hit rate, MRR, the report) is the same for all,
# so the three results are directly comparable.

import json
from pathlib import Path

import pandas as pd
from rag.embed import embed_query

TOP_K = 5   # how many chunks retrieval returns
GOLDEN_FILE = Path("eval/golden.jsonl")


# ---------- Load the golden questions ----------

def read_jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------- Search one store and get chunks back ----------

def retrieve_from_store(store, question):
    """Search any VectorStore and return the top chunks as dicts with source and page."""
    question_vector = embed_query(question)
    search_results = store.search(question_vector, TOP_K)          # [(id, score), ...]

    retrieved_chunks = [store.get_metadata(chunk_id) for chunk_id, _score in search_results]
    top_score = search_results[0][1] if search_results else 0.0
    return retrieved_chunks, top_score


# ---------- Score one question ----------

def chunk_pages(chunk):
    """Pages a chunk covers. Phase 02 chunks can span pages: stored as "3,4"."""
    if chunk.get("pages"):
        return [int(p) for p in str(chunk["pages"]).split(",")]
    return [chunk["page"]]


def is_relevant(chunk, golden_row):
    """A chunk is relevant if it comes from the right PDF and touches a right page."""
    return (chunk["source"] == golden_row["doc"]
            and any(page in golden_row["pages"] for page in chunk_pages(chunk)))


def rank_of_first_relevant(retrieved_chunks, golden_row):
    """Position (1, 2, 3...) of the first relevant chunk, or None if none was found."""
    for position, chunk in enumerate(retrieved_chunks, start=1):
        if is_relevant(chunk, golden_row):
            return position
    return None


# ---------- Run every question against one store ----------

def evaluate(store):
    golden_questions = read_jsonl(GOLDEN_FILE)
    results = []

    for golden_row in golden_questions:
        retrieved_chunks, top_score = retrieve_from_store(store, golden_row["question"])
        answerable = golden_row["doc"] is not None
        rank = rank_of_first_relevant(retrieved_chunks, golden_row) if answerable else None

        results.append({
            "id": golden_row["id"],
            "type": golden_row["type"],
            "answerable": answerable,
            "rank": rank,
            "hit": rank is not None,                     # found in the top K?
            "reciprocal_rank": 1 / rank if rank else 0,  # 1.0, 0.5, 0.33 ... or 0
            "top_score": round(float(top_score), 3),
            "wanted_pages": golden_row["pages"],
            "got_pages": [chunk["page"] for chunk in retrieved_chunks],
        })

    results_table = pd.DataFrame(results)
    results_table["rank"] = results_table["rank"].astype("Int64")   # show 1, 2 ... and <NA>, not 1.0 and NaN
    return results_table


# ---------- Print the report ----------

def report(results_table, store_name):
    answerable = results_table[results_table["answerable"]]
    unanswerable = results_table[~results_table["answerable"]]

    print(f"\n========== {store_name} ==========")
    print("\nPer question:")
    print(answerable[["id", "type", "rank", "top_score", "wanted_pages", "got_pages"]]
          .to_string(index=False))

    print(f"\nBy question type (hit@{TOP_K} and MRR):")
    by_type = answerable.groupby("type").agg(
        questions=("id", "count"),
        hit_rate=("hit", "mean"),
        mrr=("reciprocal_rank", "mean"),
    )
    print(by_type.round(2).to_string())

    hit_rate = answerable["hit"].mean()
    mrr = answerable["reciprocal_rank"].mean()
    print(f"\nOVERALL ({store_name})  hit@{TOP_K} = {hit_rate:.2f}   MRR = {mrr:.2f}")

    print("\nUnanswerable questions (top score should ideally be low):")
    print(unanswerable[["id", "top_score", "got_pages"]].to_string(index=False))
    return hit_rate, mrr
