# Failure log

**Config:** 01 naive: 800-char chunks / 150 overlap, nomic-embed-text-v1.5, top-5, NumPy brute force, Qwen3-30B-A3B
**Date:** 2026-11-13
**Baseline:** hit@5 0.82, MRR 0.76 (see [results.md](results.md))

How to decide the failed stage:
1. Answer text unreadable or missing in `chunks.jsonl`: **parsing**
2. Answer split across chunks, or buried in an unrelated chunk: **chunking**
3. Right page not in the top 5: **retrieval**
4. Right page retrieved, but wrong answer or refusal: **generation**
5. Answer cites a source that doesn't support it: **faithfulness**

| # | Question | Failed stage | Status |
|---|----------|--------------|--------|
| F1 | q016: what gets frozen | retrieval + faithfulness | open |
| F2 | q012: Wikipedia dump | retrieval (diluted chunk) | **fixed in Phase 02** (Docling + structure-aware: rank 1) |
| F3 | q013: Sequence vs Token | retrieval (wording) | open, cause confirmed |
| F4 | q021: "order of words" | retrieval (wording + table noise) | open |
| F5 | Encoder/decoder explanation | faithfulness | open |

---

## F1: q016: which parts get frozen during fine-tuning (most serious)

- **Question:** During fine-tuning, which parts of RAG get updated and which stay frozen?
- **Expected:** 2005.11401v4.pdf, p.4. The query encoder (BERT_q) and BART generator are fine-tuned; the document encoder and index stay fixed.
- **Retrieved pages:** [7, 7, 8, 7, 7] (p.4 not in the top 5)
- **Model answer:**
  > During fine-tuning, the generator part of RAG gets updated, while the retriever can be kept frozen. This is shown in the ablation studies where freezing the retriever during training still allows for improved results, indicating that the generator is updated during fine-tuning while the retriever remains unchanged in such cases [2].
- **Failed stage:** retrieval + faithfulness
- **Why (my guess):**
  - **Word mismatch.** I asked "updated / frozen"; the paper says "fine-tuning / **fixed**". The word "freeze" literally appears on p.7 in the *RAG-Frozen* ablation, so retrieval matched the word, not the meaning.
  - **Diluted chunk.** The answer chunk (`p4#c0`) starts with a loss formula and ends with RAG-Token decoding maths. The one key sentence is a small part of the chunk's vector.
  - **Wrong conclusion.** The ablation shows freezing the retriever *hurts* ("learned retrieval improves results for all tasks"). The model read it backwards and claimed the retriever is frozen. Actually part of it (the query encoder) is trained.
  - **Misleading citation.** [2] is a real passage about freezing, so the answer *looks* verified but isn't. A refusal would have been better.
- **What I checked:** The answer is complete in chunk `2005.11401v4.pdf#p4#c0`: "keep the document encoder (and index) fixed, only fine-tuning the query encoder BERTq and the BART generator". So parsing and chunk boundaries are fine; the chunk just wasn't retrieved.
- **Possible fix:**
  - Query rewriting / HyDE to match the paper's vocabulary (Phase 04)
  - Hybrid search with BM25 (Phase 03), though "frozen" vs "fixed" won't help BM25 either
  - Reranker to judge whether a chunk *answers* the question, not just shares words (Phase 05)
  - Faithfulness judge: is each claim supported by its citation? (Phase 06)
- **Status:** open

---

## F2: q012: which Wikipedia dump, how it's split, how many documents

- **Question:** Which Wikipedia dump does RAG use, how is it split, and how many documents does that produce?
- **Expected:** 2005.11401v4.pdf, p.4. December 2018 dump, disjoint 100-word chunks, 21M documents.
- **Retrieved pages:** [2, 17, 1, 9, 18]
- **Model answer:**
  > I couldn't find this in the documents.
- **Failed stage:** retrieval (the refusal itself was correct for what it was given)
- **Why (my guess):**
  - "Wikipedia" appears all over this paper (intro, appendix C on FAISS memory "for all of Wikipedia", appendix D/E on Wikipedia test sets and dumps). Many chunks are *about* Wikipedia, so they compete.
  - The answer chunk mixes three topics: the dump, the FAISS/HNSW index, and the start of section 3.1 (open-domain QA). Its vector is an average of all three.
  - **Original guess was wrong:** I thought the answer might be cut across two chunks. It isn't.
- **What I checked:** All three facts are in ONE chunk, `2005.11401v4.pdf#p4#c3`. The retrieved p.17/18 chunks mention Wikipedia and FAISS but not the dump date.
- **Possible fix:**
  - Smaller or structure-aware chunks so the dump sentence isn't mixed with index details (Phase 02)
  - Multi-part question: decompose into 3 sub-questions (Phase 04)
  - Reranker (Phase 05)
- **Status:** open

---

## F3: q013: difference between RAG-Sequence and RAG-Token

- **Question:** What's the difference between the two ways RAG combines the passages it retrieves?
- **Expected:** 2005.11401v4.pdf, p.3. RAG-Sequence uses the same document for the whole output; RAG-Token can use a different document per token.
- **Retrieved pages:** [6, 6, 7, 7, 9]
- **Model answer:**
  > I couldn't find this in the documents.
- **Failed stage:** retrieval (wording / paraphrase)
- **Why (my guess):**
  - My question says "two ways ... combines the passages". The paper says "RAG-Sequence", "RAG-Token", "marginalize over the latent documents". Almost no words in common.
  - Pages 6–7 contain the *names* RAG-Token / RAG-Seq (in results tables), so they were pulled in, but not the *definitions*. The model correctly refused.
- **What I checked:** quick_eval shows p.3 not in top 5 (`got_pages: [6, 6, 7, 7, 9]`).
- **Possible fix:**
  - Query rewriting / HyDE: write a hypothetical answer first, then search with it (Phase 04)
- **Hypothesis test (done):** asked the same thing in the paper's words, "What is the difference between RAG-Sequence and RAG-Token?"
  - Retrieved pages: **[3, 6, 6, 7, 8]**: p.3 is now **rank 1** (score 0.782)
  - Answer: correct and cited ("the same retrieved document is used to generate the entire output sequence ... RAG-Token ... draws a different latent document for each target token")
  - **Conclusion: confirmed wording problem.** Parsing, chunking and the prompt are fine; only the question's vocabulary was the issue. Query rewriting (Phase 04) should fix q013.
- **Status:** open (cause confirmed)

---

## F4: "How does the model handle the order of words?" (positional encoding)

- **Question:** How does the model handle the order of words?
- **Expected:** 1706.03762v7.pdf, p.6. It adds sinusoidal positional encodings to the input embeddings, because there is no recurrence or convolution.
- **Retrieved pages:** [1, 2, 9] (top 3, from compare_stores.py), with p.6 missing
- **Model answer:** not run through `answer()` yet
- **Failed stage:** retrieval (wording + table noise in the chunk)
- **Why (my guess):**
  - "order of words" vs the paper's "order of the sequence" / "positional encoding".
  - The answer chunk `p6#c0` starts with **Table 1** flattened into text ("Self-Attention O(n2 · d) O(1) O(1) Recurrent ..."). More than half of the chunk is table noise, and section 3.5 only begins near the end. The section continues in the next chunk.
- **What I checked:** Read chunk `1706.03762v7.pdf#p6#c0`; the table text comes first, then "3.5 Positional Encoding".
- **Possible fix:**
  - Layout-aware parsing that keeps tables separate from prose (Phase 02)
  - Section-aware chunking so 3.5 is one chunk (Phase 02)
  - Query rewriting (Phase 04)
- **Update:** added to golden.jsonl as **q021**. quick_eval confirms the miss: retrieved [1, 2, 9, 2, 3], p.6 not in top 5.
- **Status:** open

---

## F5: Encoder and decoder stacks explanation (answer with uncited claims)

- **Question:** What are Encoder and Decoder Stacks? Explain step by step
- **Expected:** 1706.03762v7.pdf, p.3 (section 3.1), plus p.5 for how attention is used.
- **Retrieved pages:** [3, 3, 5, 2, 2] (retrieval was **good**, top score 0.822)
- **Model answer (problem parts):**
  > This helps stabilize and accelerate training by mitigating vanishing gradients.

  > The masking in the self-attention sub-layer ensures that ... [4]
- **Failed stage:** faithfulness (generation)
- **Why (my guess):**
  - "Mitigating vanishing gradients" is **not in the paper** and has no citation. The model added it from its own knowledge.
  - The decoder's encoder-decoder attention detail is correct and comes from source [3] (p.5), but is **not cited**.
  - Masking is described in source [2] (p.3), but the answer cites [4] (p.2): **wrong citation**.
  - "Explain step by step" invites the model to elaborate beyond the sources.
- **What I checked:** Compared each claim against the 5 retrieved chunks by hand.
- **Possible fix:**
  - Stricter prompt: "every sentence must end with a citation; leave out anything you can't cite"
  - Faithfulness and citation-accuracy judges (Phase 06)
  - Compare with the 4B model: does it add more or less outside knowledge?
- **Status:** open

---

## Patterns so far

- **3 of 5 are wording problems** (F1, F3, F4): my question uses different words from the paper. This points to Phase 04 (query rewriting) as a likely big win.
- **3 of 5 involve chunks with mixed content** (F1, F2, F4): one key sentence diluted by formulas, tables or other topics. This points to Phase 02 (chunking).
- **Retrieval failures mostly produced correct refusals** (F2, F3). The refusal prompt works.
- **The dangerous ones are confident, cited, wrong answers** (F1, F5). The refusal check can't catch them; you need faithfulness evaluation (Phase 06).
