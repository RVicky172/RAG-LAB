# Phase 05: Post-Retrieval Processing & Reranking

**Goal:** Make sure only high-signal context reaches the generator, in the right amount and the right order.
**Time:** 1–2 weeks part-time.
**Guide sections:** Part 1 §4 (parent-child, sentence window), §6 (Reranking); Part 2 §8 (steps 6–7).

**Models this phase** (see [MODELS.md](../MODELS.md)):

| Role | Model | Why here |
|------|-------|----------|
| Reranker (default) | **BAAI/bge-reranker-v2-m3** (Python, GPU) | Strong multilingual cross-encoder; scores 50 pairs quickly on an RTX 5080. LM Studio doesn't serve rerankers, so this runs in Python. |
| Reranker (fast baseline) | BAAI/bge-reranker-base | Measures the speed vs quality trade-off |
| Late interaction (optional) | bge-m3 ColBERT vectors | See ColBERT-style scoring without a separate model |
| Context compression | Qwen3-4B-Instruct-2507 via `helper()` | Per-chunk extraction calls need a fast model |
| Generator | Qwen3-30B-A3B-Instruct-2507 | Watch how answer quality changes as context gets cleaner |

---

## Why this phase matters

Retrieval optimizes **recall** (find the right chunk somewhere in the top 50). The LLM needs **precision** (the right 3–8 chunks, no noise). More context is not better: irrelevant chunks distract the model, raise cost and latency, and hurt answers. LLMs also use the start and end of long contexts better than the middle (**"Lost in the Middle"**, Liu et al. 2023). This phase turns "50 decent candidates" into "5 excellent sources".

---

## Learning steps (in order)

### Step 1: Bi-encoders vs cross-encoders

| | Bi-encoder (embeddings) | Cross-encoder (reranker) |
|---|---|---|
| Input | query and doc encoded **separately** | query + doc **together** in one pass |
| Precompute docs? | Yes, at ingestion | No, scored at query time |
| Speed | ms over millions of docs | ~10–50 ms per pair (CPU), so dozens of docs only |
| Accuracy | good | much better (full attention between query and doc tokens) |

**Late interaction (ColBERT)** sits in between: per-token embeddings are precomputed, and query tokens are matched to doc tokens via "MaxSim". It's more accurate than a bi-encoder and cheaper than a cross-encoder, at the cost of more storage. Know it for interviews.

### Step 2: Two-stage retrieval

```
query ─► hybrid retrieve top 50 (cheap, high recall) ─► cross-encoder rerank ─► keep top 5 (high precision)
```

```python
# rag/rerank.py
from sentence_transformers import CrossEncoder
_ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")   # baseline: BAAI/bge-reranker-base

def rerank(query: str, candidates: list[dict], top_n: int = 5, min_score: float | None = None):
    scores = _ce.predict([(query, c["text"]) for c in candidates], batch_size=16)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    out = [{**c, "rerank_score": float(s)} for c, s in ranked[:top_n]]
    return [c for c in out if min_score is None or c["rerank_score"] >= min_score]
```

Managed options for later (Phase 09): Bedrock Rerank (Amazon Rerank 1.0, Cohere Rerank 3.5), Cohere, Voyage, Jina. An LLM can also rerank (listwise prompting, e.g. with Qwen3-30B-A3B) but it's slower. Try it once to see why dedicated cross-encoders win on speed.

**Tuning:**
- Candidate count (20/50/100) vs latency
- Final `top_n` (3/5/8) vs answer quality and tokens
- A **score threshold**: if even the best chunk scores low, the answer probably isn't in the corpus, so return "I don't know" early. This is a powerful anti-hallucination lever.

### Step 3: Parent-child (small-to-big) retrieval

Small chunks match precisely; large chunks give the LLM enough context. Get both:

1. At ingestion, create **parents** (a section, ~1,000–2,000 tokens) and **children** (~200–400 tokens), each child with a `parent_id`.
2. Index only children (vectors + BM25). Store parents in a key-value store or a second index.
3. At query time: retrieve and rerank children → map to parents → **deduplicate** → send parents (or the child plus neighbours) to the LLM.

```python
def expand_to_parents(children: list[dict], parent_store: dict, max_parents: int = 4):
    out, seen = [], set()
    for c in children:                        # in reranked order: best child per parent wins
        pid = c["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        p = parent_store[pid]
        out.append({**p, "page": c["page"], "match_score": c["rerank_score"]})
        if len(out) == max_parents:
            break
    return out
```

**Sentence window** variant: index single sentences; return ±N neighbouring sentences.
**Auto-merging** variant: if most children of one parent are retrieved, replace them with the parent.

Trade-offs: more tokens per request; cap parent size so it doesn't blow the token budget.

### Step 4: Diversity with MMR (Maximal Marginal Relevance)

Top results are often near-duplicates (the same paragraph in three versions of a document). MMR picks each next chunk to maximize `λ·relevance − (1−λ)·max_similarity_to_already_selected`.

> Don't confuse **MMR** (a diversity technique) with **MRR** (a ranking metric, Phase 06).

Also deduplicate exact and near-exact chunks by `content_hash` or high cosine similarity.

### Step 5: Context compression

Remove what the LLM doesn't need, inside the chunks you keep:

| Method | How | Trade-off |
|--------|-----|-----------|
| Extractive sentence filter | Score each sentence vs the query (embedding or cross-encoder); keep the top ones | Cheap; may drop needed context |
| LLM extraction | Small LLM: "copy only the sentences relevant to the question" | Accurate; one LLM call per chunk (parallelize) |
| LLMLingua / LongLLMLingua | A small LM drops low-information tokens | Big compression; text becomes less readable |
| Summarization | Summarize each chunk | Loses exact numbers and wording, so be careful with citations |

Measure: tokens saved vs faithfulness and correctness (Phase 06). Compression that drops the key number is worse than no compression.

### Step 6: Context ordering and assembly

- **Lost in the middle:** place the strongest sources first (and optionally the second strongest last); weakest in the middle. Modern long-context models are better at this than 2023 models, so measure it for your model and context length.
- **Token budget:** stop adding sources when the budget is reached (e.g. 4,000 tokens of context).
- **Number sources** `[1] [2]` with title and page, so citations are checkable.
- Group chunks from the same document together, in document order, when the question is about one document.

```python
def assemble(sources, budget_tokens=4000):
    out, used = [], 0
    for n, s in enumerate(sources, 1):
        t = len(s["text"]) // 4               # rough estimate; use the model's tokenizer for precision
        if used + t > budget_tokens:
            break
        out.append(f'[{n}] ({s["title"]}, p.{s["page"]})\n{s["text"]}')
        used += t
    return "\n\n".join(out)
```

### Step 7: Citations and grounded generation

- Ask for `[n]` after each claim; post-check in code that every cited `n` exists and was actually provided.
- Local models sometimes skip or invent citations. Measure the **citation rate** (answers with valid citations ÷ answered questions) and tighten the prompt, or add one line of examples, if it's low.
- Some cloud APIs (e.g. Claude's citations feature) return exact cited spans natively. You'll meet these in Phase 09; building `[n]` checks yourself first teaches you what they automate.
- If the reranker's top score is below the threshold, or no source survives, return the refusal without calling the LLM at all.

### Step 8: Measure (the most satisfying ablation)

```
| Config                                 | hit@5 | MRR  | faithfulness* | ctx tokens | p95 ms |
|----------------------------------------|-------|------|---------------|------------|--------|
| 04 hybrid + rewrite (top 5)            |       |      |               |            |        |
| + rerank (50 → 5)                      |       |      |               |            |        |
| + parent expansion                     |       |      |               |            |        |
| + MMR / dedupe                         |       |      |               |            |        |
| + compression                          |       |      |               |            |        |
| + score-threshold refusal              |       |      |               |            |        |
```
\* Faithfulness gets properly measured in Phase 06. For now, judge 10 answers by hand.

---

## Milestone

- `rag/rerank.py` (cross-encoder + threshold), `rag/context.py` (parent expansion, MMR/dedupe, compression, ordering, budgeted assembly)
- `rag/pipeline.py`: the full chain **transform → hybrid retrieve → rerank → expand → assemble → generate with citations**
- Ablation table showing what each step earned
- **Framework checkpoint:** re-implement your pipeline in **LlamaIndex or LangChain** (one weekend). Map each of your functions to the framework's abstraction (retriever, node postprocessor, response synthesizer). Now you can read anyone's RAG code, and you know what it does underneath.

## Exit checklist

- [ ] I can explain bi-encoder vs cross-encoder vs ColBERT.
- [ ] I know why retrieving 50 and reranking to 5 beats retrieving 5.
- [ ] I can implement parent-child retrieval with deduplication.
- [ ] I can explain "Lost in the Middle" and how I order context.
- [ ] I can say how many tokens my average prompt uses, and why.
- [ ] My system can refuse *before* calling the LLM when retrieval is weak.

## Common pitfalls

- Reranking only the top 5 (nothing left to reorder).
- Parent sections so large that 4 parents blow the context budget.
- Compression removing the exact figure the question asked for.
- Not deduplicating multiple children of the same parent.

## Interview questions

1. Why add a reranker if you already have good embeddings?
2. What's the downside of parent-child chunking?
3. What is "Lost in the Middle" and how do you counter it?
4. MMR vs MRR?
5. How do you decide how many chunks to send to the LLM?

## Resources

- Nogueira & Cho, 2019: *Passage Re-ranking with BERT*
- Khattab & Zaharia, 2020: *ColBERT*
- Liu et al., 2023: *Lost in the Middle*
- Jiang et al., 2023: *LLMLingua* / *LongLLMLingua*
- Carbonell & Goldstein, 1998: *MMR*
- Guide: Part 1 §4, §6; Interview Q7, Q15

## Next

→ [Phase 06: Evaluation & Observability](../06-evaluation-observability/README.md)
