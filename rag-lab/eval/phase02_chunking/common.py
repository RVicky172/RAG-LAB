# eval/phase02_chunking/common.py
# Shared code for the Phase 02 chunker ablation, used by:
#     eval_numpy.py    eval_faiss.py    eval_chroma.py   (same folder)
#
# For each chunk file in data/chunks/:
#   1. load the chunks
#   2. embed them ONCE (cached in data/cache/embeddings/, shared by all 3 stores)
#   3. let the calling script put them in its store
#   4. score the golden set with eval_core.evaluate (hit@5, MRR)
#   5. append the result to eval/phase02_chunking/runs.jsonl
#
# Only the store differs between the three scripts, so their numbers are
# directly comparable (and should match: all three do cosine similarity).

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
from eval.eval_core import TOP_K, evaluate, read_jsonl, report
from rag.embed import MODEL as EMBED_MODEL
from rag.embed import embed_docs

from rag.paths import CHUNKS as CHUNKS_DIR
from rag.paths import EMBEDDINGS_CACHE as VECTORS_DIR
RUNS_FILE = Path("eval/phase02_chunking/runs.jsonl")


# ---------- 1. Which chunk files ----------

def chunk_configs(names):
    """All chunk files (skipping partial "-limit" timing runs), or the ones named."""
    if names:
        return names
    return sorted(p.stem for p in CHUNKS_DIR.glob("*.jsonl") if "-limit" not in p.stem)


# ---------- 2. Embed once, cache on disk ----------

def load_embedded_chunks(config):
    """Return (chunks, vectors). Vectors are cached per chunk file + embedding model."""
    chunks = read_jsonl(CHUNKS_DIR / f"{config}.jsonl")
    texts = [chunk["text"] for chunk in chunks]

    fingerprint = hashlib.sha256((EMBED_MODEL + "\n".join(texts)).encode()).hexdigest()[:12]
    model_tag = EMBED_MODEL.split("/")[-1]
    cache = VECTORS_DIR / f"{config}.{model_tag}.{fingerprint}.npy"

    if cache.exists():
        vectors = np.load(cache)
    else:
        print(f"  embedding {len(texts)} chunks with {EMBED_MODEL} ...", flush=True)
        vectors = embed_docs(texts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return chunks, vectors


def store_metadata(chunk):
    """Flat metadata every store accepts (Chroma allows only str/int/float/bool)."""
    return {
        "source": chunk["source"],
        "page": chunk["page"],
        "pages": ",".join(str(p) for p in chunk["pages"]),
        "element_type": chunk["element_type"],
        "section_path": chunk["section_path"],
    }


# ---------- 3. Score and record ----------

def summarize(results_table):
    answerable = results_table[results_table["answerable"]]
    by_type = answerable.groupby("type").agg(hit=("hit", "mean"), mrr=("reciprocal_rank", "mean"))
    return {
        "hit_at_k": round(float(answerable["hit"].mean()), 3),
        "mrr": round(float(answerable["reciprocal_rank"].mean()), 3),
        "missed": answerable.loc[~answerable["hit"], "id"].tolist(),
        "by_type": {t: {"hit": round(r.hit, 2), "mrr": round(r.mrr, 2)} for t, r in by_type.iterrows()},
    }


def record(store_name, config, chunks, summary):
    golden = read_jsonl(Path("eval/golden.jsonl"))
    row = {
        "date": date.today().isoformat(),
        "store": store_name,
        "config": config,
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
        "golden_questions": len(golden),
        "n_chunks": len(chunks),
        "avg_tokens": round(sum(c["n_tokens"] for c in chunks) / len(chunks)),
        **summary,
    }
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def print_table(rows):
    print(f"\n{'config':<55}{'chunks':>7}{'avg tok':>8}{'hit@' + str(TOP_K):>7}{'MRR':>7}  missed")
    for r in rows:
        print(f"{r['config']:<55}{r['n_chunks']:>7}{r['avg_tokens']:>8}{r['hit_at_k']:>7.2f}"
              f"{r['mrr']:>7.2f}  {', '.join(r['missed']) or '-'}")


# ---------- 4. The loop every script runs ----------

def run(store_name, build_store):
    """build_store(config, chunks, vectors) -> a filled VectorStore."""
    parser = argparse.ArgumentParser(description=f"Chunker ablation on {store_name}")
    parser.add_argument("configs", nargs="*", help="chunk file names (default: all)")
    parser.add_argument("--verbose", action="store_true", help="per-question report")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    for config in chunk_configs(args.configs):
        print(f"[{store_name}] {config}", flush=True)
        chunks, vectors = load_embedded_chunks(config)
        store = build_store(config, chunks, vectors)
        results_table = evaluate(store)
        if args.verbose:
            report(results_table, f"{store_name}: {config}")
        rows.append(record(store_name, config, chunks, summarize(results_table)))

    print(f"\n===== {store_name}  (embeddings: {EMBED_MODEL}, golden: eval/golden.jsonl) =====")
    print_table(rows)
    print(f"\nAppended to {RUNS_FILE}")
