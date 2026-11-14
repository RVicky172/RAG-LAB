# Phase 02 results: parser + chunker ablation


**Setup:** golden **v2** (21 q, 18 answerable) · top-5 · embeddings nomic-embed-text-v1.5 · corpus **4 PDFs, 60 pages** (added 2307.03172 *Lost in the Middle* and 2309.15217 *RAGAS*, both two-column).
**Scripts:** `uv run python -m eval.phase02_chunking.eval_numpy` (also `eval_faiss`, `eval_chroma`; shared code in [common.py](common.py); every run is appended to [runs.jsonl](runs.jsonl)).
**Relevance:** a chunk counts if it's from the right PDF and **any** of its pages is a golden page (chunks can now span pages).

> Not comparable with the [Phase 01 results](../phase01_naive/results.md): the corpus doubled (more distractors) and the chunk sizes changed. The `pypdf-recursive` row is the new baseline.

| Config | Parser | #chunks | avg tok | hit@5 | MRR | Missed |
|--------|--------|---------|---------|-------|-----|--------|
| 02 baseline: recursive 512/64 | pypdf | 117 | 487 | 0.72 | 0.69 | q011, q012, q013, q016, q021 |
| 02a recursive 512/64 | docling | 117 | 432 | 0.83 | 0.64 | q013, q016, q021 |
| 02c semantic p95, cap 512 | docling | 196 | 251 | 0.83 | 0.66 | q013, q016, q021 |
| 02b structure-aware + section path, 512/64 | docling | 202 | 262 | 0.83 | 0.74 | q013, q016, q021 |
| 02d structure-aware + contextual (Qwen3-4B) | docling | 202 | 332* | 0.83 | **0.77** | q013, q016, q021 |

| 02e structure-aware, 400/64 (control for 02f) | docling | 232 | 233 | **0.89** | 0.75 | q013, q021 |
| 02f structure-aware 400/64 + contextual (Qwen3-4B) | docling | 232 | 302 | 0.83 | 0.74 | q013, q016, q021 |

\* 43 of 202 contextual chunks in 02d exceed 512 tokens (context adds ~70 tokens), so 02d is not size-fair. 02f fixes that: base chunks ≤ 400, enriched chunks ≤ 477 tokens.

**Same numbers on all three stores** (NumPy exact, FAISS IndexFlatIP exact, Chroma HNSW approximate). Expected: at ~200 vectors HNSW finds the true neighbours. The store choice matters for speed and features at scale (Phase 03), not for quality here.

### hit@5 / MRR by question type

| Config | fact (3) | identifier (3) | multi-part (4) | paraphrased (4) | table (4) |
|--------|----------|----------------|----------------|-----------------|-----------|
| pypdf recursive | 1.00 / 0.78 | 0.67 / 0.67 | 0.75 / 0.75 | 0.25 / 0.25 | 1.00 / 1.00 |
| docling recursive | 1.00 / 0.67 | 1.00 / 0.75 | 1.00 / 0.71 | 0.25 / 0.25 | 1.00 / 0.88 |
| docling semantic | 1.00 / 0.67 | 1.00 / 0.78 | 1.00 / 0.80 | 0.25 / 0.25 | 1.00 / 0.83 |
| docling structure | 1.00 / 0.78 | 1.00 / 1.00 | 1.00 / 0.88 | 0.25 / 0.25 | 1.00 / 0.88 |
| docling structure + contextual | 1.00 / 0.78 | 1.00 / 1.00 | 1.00 / 0.88 | 0.25 / 0.25 | 1.00 / 1.00 |
| docling structure 400 | 1.00 / 0.83 | 1.00 / 1.00 | 1.00 / 0.88 | 0.50 / 0.30 | 1.00 / 0.83 |
| docling structure 400 + contextual | 1.00 / 0.78 | 1.00 / 1.00 | 1.00 / 0.88 | 0.25 / 0.25 | 1.00 / 0.88 |

### What the numbers say

1. **The parser matters most (hit@5 0.72 → 0.83).** Docling fixed q011 (BART-large params, rank miss → rank 1 with structure) and **q012, failure F2 "diluted chunk"**: the Wikipedia-dump answer now sits in a focused chunk at rank 1.
2. **Chunking changes the ranking, not the hits (MRR 0.64 → 0.74 → 0.77).** Same 15/18 found by every Docling config; structure-aware puts the right chunk higher. Identifier questions go 0.75 → 1.00 MRR: the section path in the chunk ("RAG > 2.2 Generator") helps short lookups.
3. **Semantic ≈ recursive (0.66 vs 0.64).** On well-structured papers it adds embedding cost for no gain, as expected; it also crosses section boundaries.
4. **Contextual retrieval: small gain (0.74 → 0.77, table MRR 0.88 → 1.00)** for 16 minutes of LLM time on 60 pages. Promising but within noise on 18 questions.
5. **The 3 remaining misses are all paraphrased questions** (q013, q016, q021, same as Phase 01 F1/F3/F4). No chunker fixes word mismatch ("frozen" vs "fixed"); that's a query-side problem: hybrid search (Phase 03) and query rewriting / HyDE (Phase 04).

### Size-fair re-test of contextual retrieval (02e vs 02f)

Contextual at 512 (02d) changed two things at once: it added a context **and** made chunks bigger (up to 582 tokens). The re-test separates them: same base chunker at 400 tokens, with (02f) and without (02e) context. Context cost: 94 new LLM calls, 138 reused from the cache (7.7 min).

| | hit@5 | MRR |
|---|---|---|
| structure 512 → structure 400 (smaller chunks only) | 0.83 → 0.89 | 0.74 → 0.75 |
| structure 400 → + contextual (context only) | 0.89 → 0.83 | 0.75 → 0.74 |

6. **At equal size, contextual retrieval gives no gain here (0.75 → 0.74 MRR) and loses q016.** The 02d gain (0.74 → 0.77) doesn't reproduce, so it was most likely noise and/or the bigger chunks, not the context.
7. **Why it may not help on this corpus (hypothesis, not proven):** every context starts with the paper title ("This chunk is from *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*..."). That makes all chunks of one paper more similar to each other, so it helps choose the *paper* but not the *passage*. Our golden questions already land in the right paper; the hard part is the passage. Contextual retrieval should matter more for chunks that are ambiguous on their own ("revenue grew 12%") and in a corpus of many similar documents.
8. **structure-400's extra hit is borderline:** q016 is found at **rank 5, score 0.707**, the very edge of the top 5; adding context lifts other chunks a little more and pushes it out. One borderline question is not evidence that 400 beats 512. q016 is still a word-mismatch problem ("frozen" vs "fixed").

### Caveats
- **18 answerable questions:** one question = 0.056 hit@5. Differences under ~0.05 MRR are noise.
- **The golden set only covers the 2 original papers.** The two new two-column papers (where Docling helped most visually) and their tables/figures aren't tested yet.
- Anthropic's published gains combine contextual embeddings **with contextual BM25** (and a reranker). We've only tested the embedding half; re-test after hybrid search (Phase 03) and reranking (Phase 05).

### Decision
Use **Docling + structure-aware chunking with the section path prepended**, at **400 tokens / 64 overlap** (best hit@5 0.89 and MRR 0.75 without any LLM cost; smaller chunks also leave room for a context later). 400 vs 512 is within noise, so re-check the size (256 / 400 / 512 / 1024) when the golden set has 40 questions. **Contextual retrieval: not adopted for now.** At equal chunk size it adds 8–16 min of LLM time per corpus run for no measured gain; revisit it with BM25 (Phase 03), a reranker (Phase 05) and the 40-question golden set. Drop semantic chunking for this corpus.

---

## Golden v3 re-run (2026-11-14)

**Golden v3:** 42 questions, **38 answerable** (+21: 14 on *Lost in the Middle*, 6 on *RAGAS*, 1 unanswerable; adds 5 table and 4 figure questions). One question = **0.026** hit@5 (was 0.056). Same corpus, chunks, embeddings and top-5 as above. **Don't compare v3 rows with the v2 rows above.** Identical numbers on NumPy, FAISS and Chroma.

| Config | #chunks | hit@5 | MRR | Missed |
|--------|---------|-------|-----|--------|
| 02 baseline: pypdf recursive 512/64 | 117 | 0.82 | 0.67 | q011, q012, q013, q016, q021, q030, q038 |
| 02a docling recursive 512/64 | 117 | 0.84 | 0.67 | q013, q016, q021, q030, q038, q040 |
| 02c docling semantic p95 | 196 | 0.87 | 0.68 | q013, q016, q021, q030, q036 |
| 02b docling structure 512/64 | 202 | 0.90 | 0.72 | q013, q016, q021, q023 |
| 02d structure 512 + contextual | 202 | 0.87 | 0.77 | q013, q016, q021, q022, q023 |
| **02e docling structure 400/64** | 232 | **0.92** | **0.78** | q013, q021, q023 |
| 02f structure 400 + contextual | 232 | 0.87 | 0.72 | q013, q016, q021, q023, q038 |

### By question type (hit@5 / MRR)

| Config | fact (4) | identifier (6) | multi-part (7) | paraphrased (8) | table (9) | figure (4) |
|--------|----------|----------------|----------------|-----------------|-----------|------------|
| pypdf recursive | 1.00 / 0.65 | 0.67 / 0.56 | 0.86 / 0.69 | 0.50 / 0.35 | 1.00 / 0.87 | 1.00 / 1.00 |
| docling recursive | 1.00 / 0.58 | 0.83 / 0.60 | 0.86 / 0.60 | 0.50 / 0.44 | 1.00 / 0.94 | 1.00 / 0.88 |
| docling semantic | 1.00 / 0.58 | 1.00 / 0.67 | 0.86 / 0.65 | 0.50 / 0.44 | 1.00 / 0.87 | 1.00 / 0.88 |
| structure 512 | 1.00 / 0.65 | 0.83 / 0.75 | 1.00 / 0.64 | 0.62 / 0.56 | 1.00 / 0.89 | 1.00 / 0.83 |
| structure 512 + ctx | 1.00 / 0.83 | 0.83 / 0.70 | 0.86 / 0.68 | 0.62 / 0.62 | 1.00 / 0.94 | 1.00 / 0.83 |
| **structure 400** | 1.00 / 0.88 | 0.83 / 0.70 | 1.00 / 0.74 | **0.75 / 0.59** | 1.00 / 0.87 | 1.00 / 1.00 |
| structure 400 + ctx | 1.00 / 0.71 | 0.67 / 0.67 | 1.00 / 0.72 | 0.62 / 0.54 | 1.00 / 0.87 | 1.00 / 0.88 |

### By paper (hit@5 / MRR)

| Config | original 2 papers (18 q) | new two-column papers (20 q) |
|--------|--------------------------|------------------------------|
| pypdf recursive | 0.72 / 0.69 | 0.90 / 0.65 |
| docling recursive | 0.83 / 0.64 | 0.85 / 0.70 |
| structure 512 | 0.83 / 0.74 | 0.95 / 0.70 |
| **structure 400** | 0.89 / 0.75 | **0.95 / 0.79** |
| structure 400 + ctx | 0.83 / 0.74 | 0.90 / 0.71 |

(The "original 2 papers" column reproduces the v2 numbers exactly, a good consistency check.)

### What v3 says

9. **The ranking of configs holds with twice the questions.** structure-400 is best on both golden versions (v3: 0.92 / 0.78, misses only q013, q021, q023). 400 beats 512 on hit@5 *and* MRR on both v2 and v3 (MRR +0.06 on v3), so the 400 decision is now backed by two measurements, not one borderline question.
10. **Docling helps most on the two-column papers' ranking:** MRR 0.65 (pypdf) → 0.79 (structure-400) on the new papers. pypdf still *finds* most answers there (0.90 hit@5) because these PDFs happen to store text column by column (see lab2a shoot-out), but its chunks rank lower.
11. **Contextual retrieval is inconsistent:** +0.05 MRR at 512 (0.72 → 0.77) but −0.06 at 400 (0.78 → 0.72). An effect that flips sign with chunk size is not a reliable gain. Still **not adopted**.
12. **Tables and figures: 1.00 hit@5 for every config.** Captions name the content ("Table 1: Closed-book and oracle accuracy...") and the questions are also answerable from the text, so they don't yet test chart-only content; Step 5 (vision captions) needs chart-only questions to be measurable.
13. **Paraphrased questions are the weak spot** (0.50–0.75 hit@5 across configs), as in Phase 01.
14. **New failure q023 (Contriever):** "Which retrieval system is used to select the distractor documents?" retrieves pages 14–15 (Appendices A–C, all about *distractor documents*) instead of p.3, where Contriever is named. The question's key phrase matches a whole appendix about the topic, not the passage with the answer. Candidate fixes: a reranker that reads question + passage together (Phase 05), metadata/section boosting.

### Decision (v3, supersedes the one above)
**Docling + structure-aware 400/64 with section path** (`docling-structure-400-64`): hit@5 **0.92**, MRR **0.78** on golden v3, best on every view (overall, both paper groups, paraphrased). Contextual retrieval and semantic chunking not adopted. Next: chunk sizes 256 / 400 / 512 and the embedding model (nomic vs Qwen3-Embedding-0.6B) on this config; chart-only questions before Step 5.

---

## Golden v4: chart questions (2026-11-14)

**Golden v4:** 57 questions, **53 answerable** = v3 + 15 `chart` questions (q043–q057), 5 each on Attention, RAG and Lost in the Middle (RAGAS has no figures). Each asks for something read off a figure: a plotted value ("about 45% at the 200th position"), a diagram label ("Mask (opt.)"), or an example drawn in a figure. Written by looking at the crops in `data/figures/`; values read from plots are approximate. No vision captions yet: this is the **before** measurement for Step 5. Identical on NumPy, FAISS and Chroma.

| Config | hit@5 (all 53) | MRR | chart (15) hit / MRR | non-chart (38) hit / MRR | chart misses |
|--------|----------------|-----|----------------------|--------------------------|--------------|
| pypdf recursive 512 | 0.85 | 0.71 | **0.93 / 0.82** | 0.82 / 0.67 | q052 |
| docling recursive 512 | 0.83 | 0.69 | 0.80 / 0.72 | 0.84 / 0.67 | q048, q055, q056 |
| docling semantic | 0.87 | 0.69 | 0.87 / 0.73 | 0.87 / 0.68 | q048, q055 |
| docling structure 512 | 0.89 | 0.73 | 0.87 / 0.76 | 0.89 / 0.72 | q048, q055 |
| structure 512 + ctx | 0.87 | 0.76 | 0.87 / 0.73 | 0.87 / 0.77 | q048, q055 |
| **docling structure 400** | **0.91** | **0.76** | 0.87 / 0.73 | **0.92 / 0.77** | q048, q055 |
| structure 400 + ctx | 0.87 | 0.71 | 0.87 / 0.69 | 0.87 / 0.72 | q048, q055 |

### What the chart questions show

15. **Page-level hit@5 overstates chart performance (0.87 with no vision captions at all).** Our relevance rule only checks that a retrieved chunk is on the right *page*. For a chart question, the figure's caption chunk or the text that discusses the figure sits on that page and gets retrieved, but it does **not** contain the answer ("about 45%", "Law and application"). The retriever is right about *where*; the context still can't answer. So Step 5 must be measured at the **answer level** (does the retrieved context contain the answer / is the generated answer correct), not with page hit@5. That's Phase 06's generation metrics, or a simpler "evidence string in context" check.
16. **pypdf does best on charts (0.93 / 0.82).** The "word salad" pypdf extracts from figures (axis labels, legends, diagram labels like "Mask (opt.)", the words in the attention plots) lands in its chunks and matches chart questions lexically. Docling deliberately drops that text and keeps only the caption, so it loses q048 (the RAG overview figure's example "Barack Obama was born in Hawaii." / "supports" exists only as figure text). The noise that hurts prose questions helps diagram questions: a vision caption should give Docling that text back, cleanly.
17. **q055 (Contriever recall, Figure 11)** is missed by every Docling config: the page-10 chunk is just the figure caption, and the discussion is on page 9.
18. **Decision unchanged:** structure-400 is still best overall (0.91 / 0.76) and on non-chart questions (0.92 / 0.77).

### Plan for Step 5 (vision captions)
Caption every figure with Qwen3.6-35B-A3B (axis values, labels, example text), add the caption to the figure chunk, and measure (a) chart hit@5 (expect q048, q055 to be found), and (b) the answer level: whether the top-5 context contains the chart answer, before vs after.

---

## Step 5: vision captions (2026-11-14)

**What changed (one variable):** every figure chunk now holds the printed caption **plus** a 3-5 sentence description written by **Qwen3.6-35B-A3B** looking at the figure crop (`rag.caption`, cached in `data/cache/captions.jsonl`; `rag.chunk structure --captions`). The model also gets the section path, the printed caption and nearby text (paragraphs that mention "Figure N" first, then the same section), and is told to read labels and values from the image. Same parser, chunker (400/64), embedding model (nomic-v1.5) and golden set (v4) as the baseline.

**New metric, fig@5** (chart questions only): is the **figure chunk itself**, on the right page, in the top 5? Page-level hit@5 counts any chunk from that page (finding 15), so it can't show whether the answer is in the context; fig@5 can.

| Config | chunks | hit@5 (53) | MRR | chart hit / MRR | **chart fig@5** | non-chart | misses |
|--------|--------|-----------|-----|-----------------|-----------------|-----------|--------|
| docling structure 400/64 | 232 | 0.91 | 0.76 | 0.87 / 0.73 | 0.53 (8/15) | unchanged | q013, q021, q023, q048, q055 |
| **+ vision captions** (`-vcap`) | 235 | **0.92** | **0.80** | **0.93 / 0.90** | **0.93 (14/15)** | unchanged (±0.02 MRR) | q013, q021, q023, q048 |

Identical on NumPy (exact). Captioning cost: 26 figures, 275 s (~11 s/figure), run once.

### What the captions changed

19. **The figure chunk is now what gets retrieved for chart questions: fig@5 0.53 → 0.93.** Before, 7 of 15 "hits" were text from the right page that did not contain the answer. Chart MRR 0.73 → 0.90: the figure chunk is rank 1 for 11 of 15.
20. **The answers are in the descriptions.** Checked by hand: Figure 1 (Attention) "feeds into a Linear layer and then a Softmax" (q043); "a pink Mask (opt.) block" (q044); "combined in a yellow box labeled Concat before... a final linear projection" (q045). None of these words are in the printed captions.
21. **Captionless figures are no longer dropped (23 → 26 figure chunks).** Docling split Attention Figure 2 into two pictures; the left half (`p004-fig1`, scaled dot-product attention) has no printed caption, so the old chunker skipped it. q044's answer lives only there.
22. **q055 fixed** (Contriever recall, Figure 11): the page-10 figure chunk now describes the plot and ranks 1.
23. **q048 still missed: descriptions are lossy.** The RAG Figure 1 description explains the architecture (query encoder, MIPS, generator) but leaves out the example text drawn in the figure ("Barack Obama was born in Hawaii." → "supports"). The model summarised; the question asks for a detail it dropped. Possible fix (separate experiment): ask it to also transcribe all visible text.
24. **Non-chart questions unchanged**, as expected: only figure chunks changed. Overall MRR +0.04 comes entirely from charts.

### Gotcha: thinking models and empty captions
The first full run cached **14 empty captions** out of 26. Qwen3.6 is a thinking model: on busy line plots it spent all 2000 output tokens reasoning (`finish_reason: length`, 1966 reasoning tokens) and never wrote the answer. Fixes in `rag.caption`: send `reasoning_effort: "none"` (LM Studio honours it; `chat_template_kwargs.enable_thinking` and `/no_think` were ignored), which is also 3-6x faster; raise an error instead of caching an empty or truncated reply; the reasoning setting is part of the cache key. All 26 were re-captioned with one setting so the comparison has one variable. The thinking-on cache is kept as `captions.thinking.jsonl`. Also: load the vision model with `--context-length 8192 --parallel 1`; the default 262K context x 4 slots filled VRAM and the first request returned HTTP 400.

### Decision
**Adopt vision captions:** `docling-structure-400-64-vcap` (hit@5 0.92, MRR 0.80, chart fig@5 0.93) is the new Phase 02 best config. Still to do: an answer-level check (generate from the top 5 and judge, Phase 06), and the "transcribe visible text" prompt for q048-style questions.

---

## Step 6: embedding model and chunk size (2026-11-14)

Golden v4 (53 answerable), NumPy exact search, Docling. hit@5 / MRR; fig@5 = chart questions where the figure chunk itself is in the top 5.

### 6a. Embedding model

| Embedding model (dims) | structure-400 **+ captions** | structure-400 | structure-512 | docling recursive-512 (no header) | pypdf recursive-512 |
|------------------------|------------------------------|---------------|---------------|-----------------------------------|---------------------|
| **nomic-embed-v1.5** (768) | **0.92 / 0.80**, fig 0.93 | 0.91 / 0.76 | 0.89 / 0.73 | 0.83 / 0.69 | 0.85 / 0.71 |
| Qwen3-Embedding-0.6B (1024), our instruction | 0.70 / 0.58 | 0.72 / 0.56 | | 0.76 / 0.54 | 0.72 / 0.61 |
| Qwen3-Embedding-0.6B (1024), model-card instruction | 0.85 / 0.74, fig 0.87 | 0.77 / 0.70 | 0.81 / 0.75 | 0.87 / 0.70 | 0.81 / 0.70 |
| Qwen3-Embedding-4B (2560), model-card instruction | 0.91 / 0.80, fig 0.87 | **0.93** / 0.76 | 0.91 / 0.76 | **0.93 / 0.78** | 0.87 / 0.68 |

Misses on structure-400 + captions: nomic q013, q021, q023, q048 · Qwen3-4B q014, q016, q038, q048, q056.

25. **The query instruction mattered more than the model.** Qwen3 embedders are instruction-tuned: queries get `Instruct: <task>\nQuery:<question>`. Our own wording ("Given a question, retrieve passages that answer it") scored **0.70**; the model card's exact wording ("Given a web search query, retrieve relevant passages that answer the query") **0.85** on the same chunks, +15 points from one sentence. Diagnosis path: first suspected the LM Studio GGUF build (last-token pooling), so ran the same model through sentence-transformers on the GPU: **same 0.72**, runtime ruled out; then looked at what was retrieved (finding 26); then varied only the instruction. Fixed in `rag/embed.py`. Lesson: copy prefixes/instructions verbatim from the model card, and test them.
26. **Qwen3-0.6B is pulled to the wrong paper by the section-path header.** Its misses were Lost-in-the-Middle questions that don't name the paper ("20 retrieved documents", "multi-document QA") retrieving RAG-paper chunks: every one of those starts with "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks > ...", which is on-topic for any retrieval question. The header that helps nomic (+0.08 hit@5 over recursive) *hurts* Qwen3-0.6B (0.85 with header vs 0.87 without). Same chunks, opposite effect: a chunking choice is only "best" for a given embedder.
27. **Qwen3-4B ties nomic on the chosen config (0.91 / 0.80 vs 0.92 / 0.80)** and is best on header-less chunks (0.93 / 0.78). It solves all three of nomic's hard misses (q013, q021 paraphrase; q023 topic-word trap) but misses q014, q016, q038, q056 instead. Different models fail on different questions: a hint for hybrid search / fusion (Phase 03).
28. **Cost:** nomic 84 MB / 768 dims; Qwen3-4B 2.5 GB VRAM / 2560 dims (3.3x the index size, slower to embed). At a tie, the small one wins.

### 6b. Chunk size (structure-aware + captions, nomic)

| Size / overlap | chunks | avg tok | hit@5 | MRR | chart fig@5 | non-chart misses |
|----------------|--------|---------|-------|-----|-------------|------------------|
| 256 / 64 | 335 | 194 | 0.92 | 0.79 | 0.87 | q013, q016, q021 |
| **400 / 64** | 235 | 253 | **0.92** | **0.80** | **0.93** | q013, q021, q023 |
| 512 / 64 | 205 | 284 | 0.92 | 0.77 | 0.93 | q013, q016, q021 |
| 1024 / 64 | 177 | 324 | 0.85 | 0.74 | 0.93 | q013, q016, q021, q023, q036, q037, q038 |

29. **Flat from 256 to 512, worse at 1024.** 400 has the best MRR and fig@5, so the earlier 400 decision holds with captions. At 1024 hit@5 drops 7 points: a chunk holding several facts gets a blurred vector (q036-q038, RAGAS metric definitions, lose to bigger mixed chunks).
30. **Average size grows much less than the limit** (194 → 324 tokens for a 4x limit): the structure chunker never crosses a section, so a bigger limit only merges whole short sections. At 256, 18 chunks exceed the limit; they are figure chunks (header + caption + ~130-word description), which are never split.

### Decision (Step 6)
**Keep `docling-structure-400-64-vcap` with nomic-embed-v1.5** (hit@5 0.92, MRR 0.80, chart fig@5 0.93). Qwen3-4B ties at 30x the size; revisit it in Phase 03 as a second retriever, because it catches the paraphrase misses nomic can't.

---

## Phase 02 wrap-up: lessons

1. **Parsing sets the ceiling.** Docling's reading order, tables and figure crops beat pypdf's text layer on two-column papers; nothing downstream can recover what parsing loses.
2. **Structure beats cleverness.** Section-aware chunks with the section path in the text beat semantic chunking and fixed windows, for free.
3. **Chunk size has a broad sweet spot** (256–512 here); very large chunks blur the vector. Tune it on your own golden set.
4. **Figures need vision.** The answer to a chart question is in the pixels; a vision-LLM description made the figure chunk retrievable (fig@5 0.53 → 0.93). Descriptions are lossy summaries, so ask for visible text explicitly when it matters (q048).
5. **The metric must measure the thing.** Page-level hit@5 said charts were fine (0.87) when the retrieved text didn't contain the answer; fig@5 showed the real gap.
6. **Embedding models are not interchangeable, and setup matters as much as the model.** One wrong instruction sentence cost Qwen3 15 points; the same section header helped nomic and hurt Qwen3. A bigger model (4B) tied, but failed on *different* questions.
7. **Verify outputs, fail loudly.** A thinking model silently produced 14 empty captions that were cached as successes; only checking the output caught it.
8. **Change one thing, measure, keep or revert.** Contextual retrieval looked good at 512 and bad at 400: one measurement is not a result.
