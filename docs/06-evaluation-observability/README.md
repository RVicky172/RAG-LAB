# Phase 06: Evaluation & Observability

**Goal:** Move from vibes-based testing to quantitative, repeatable measurement, and see inside every request.
**Time:** about 2 weeks part-time.
**Guide sections:** Part 2 §9 (RAG evaluation), §10 (Observability); Interview Q22–Q26.

**Models this phase** (see [MODELS.md](../MODELS.md)):

| Role | Model | Why here |
|------|-------|----------|
| System under test (generator) | Qwen3-30B-A3B-Instruct-2507 | What you're measuring |
| **LLM judge** | **gpt-oss-20b** (LM Studio) | A **different model family** from the Qwen generator, which reduces self-preference bias. A reasoning model with good JSON output, and it fits in 16 GB VRAM. Alternatives: Gemma 3 27B, Mistral Small 3.2 24B. |
| Synthetic question generation | Qwen3-30B-A3B (quality) or Qwen3-4B (volume) | Free locally, so generate many and keep only the best after human review |
| Judge check | Qwen3-30B-A3B as a second judge | Compare the two judges' agreement with your human labels; the gap shows how much judge choice matters |

> **VRAM tip:** run generation for the whole golden set first and save the answers to JSONL. Then unload the generator, load the judge, and grade. Two big models don't need to share 16 GB at the same time.

> **Why this comes before Advanced Paradigms:** GraphRAG and agentic RAG are expensive. You should only adopt them when your evaluation proves a failure class (multi-hop, global questions) that simpler fixes can't solve. You've been collecting hit@5 and MRR since Phase 01; this phase makes evaluation complete, automatic and trustworthy.

---

## Why this phase matters

Every RAG change (chunk size, embedding model, prompt, reranker, LLM version) can improve some questions and silently break others. Without an automated eval you'll ship regressions. In interviews, **"how do you evaluate it?"** is the question that separates engineers who have run RAG in production from those who followed a tutorial.

---

## Learning steps (in order)

### Step 1: Decoupled evaluation (the RAG triad)

Evaluate **retrieval** and **generation** separately, so a wrong answer points to the stage to fix.

```
             Context relevance                    Groundedness / Faithfulness
  Question ───────────────────► Retrieved context ────────────────────────────► Answer
      │                                                                           ▲
      └──────────────────────── Answer relevance ─────────────────────────────────┘
```

### Step 2: Retrieval metrics (did we fetch the right chunks?)

| Metric | Definition | Tells you |
|--------|------------|-----------|
| Hit rate@k | 1 if any relevant item is in the top k; averaged | Found at all (single-answer questions) |
| Recall@k | Fraction of *all* relevant items in the top k | Coverage for multi-evidence questions |
| Precision@k | Fraction of the top k that is relevant | Noise sent to the LLM |
| MRR | Mean of 1 / rank of the first relevant item | How high the right chunk ranks |
| nDCG@k | Rank-weighted gain with graded relevance (0/1/2) | Ranking quality with several relevant items |
| Context precision / recall (RAGAS) | LLM-judged relevance and coverage of the context | Quality without labelled chunks |

**Worked example (k=5):** Q1's answer is at rank 1 (RR=1.0), Q2's at rank 3 (0.33), and Q3's is missing (0). Hit@5 = 2/3 = 0.67. MRR = (1 + 0.33 + 0)/3 = 0.44.

**Diagnosis:** low recall → parsing, chunking, embeddings, hybrid or query rewriting. Good recall but low MRR → reranking.

```python
import math
def ndcg_at_k(ranked_rel: list[int], k: int) -> float:   # ranked_rel: graded relevance per rank
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(ranked_rel[:k]))
    ideal = sorted(ranked_rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0
```

### Step 3: Generation metrics (is the answer good?)

| Metric | Question it answers | How it's scored |
|--------|---------------------|-----------------|
| **Faithfulness / groundedness** | Is every claim supported by the retrieved context? (the hallucination metric) | LLM judge splits the answer into claims and checks each against the context |
| **Answer relevance** | Does it address the question? | LLM judge, or embedding similarity of generated questions |
| Correctness | Does it match the reference answer? | LLM judge vs reference; exact match for numbers |
| Completeness | Does it cover all parts? | LLM judge with rubric |
| Citation accuracy | Does each cited source support its claim? | LLM judge per citation |
| Refusal behaviour | Does it say "I don't know" for unanswerable questions, and *not* for answerable ones? | Exact match on the refusal string |

### Step 4: LLM-as-a-judge, done properly

Judges are biased toward longer answers, sensitive to order, and may favour their own model family.
- Use **explicit rubrics** with binary or 1–3 scales ("1 = every claim is supported by a numbered source").
- Ask for reasoning *before* the score; use **structured output** (JSON schema) for the verdict.
- Use a different model family from the generator (here: gpt-oss-20b judging Qwen answers).
- Use LM Studio's `response_format` JSON schema so every verdict parses.
- **Calibrate:** label 30–50 examples yourself; measure judge–human agreement (accuracy or Cohen's κ). Re-check whenever the judge model changes.

```python
JUDGE = """You are grading a RAG answer for faithfulness.
<context>{context}</context>
<answer>{answer}</answer>
Step 1: list each factual claim in the answer.
Step 2: for each claim, say SUPPORTED or UNSUPPORTED by the context, quoting evidence.
Step 3: output JSON: {{"claims": n, "supported": m, "faithfulness": m/n}}"""
```

### Step 5: Golden and synthetic test datasets

You have 55+ hand-written questions. Scale to **100–200**:
1. **Synthetic generation:** sample chunks (stratified by document and element type); an LLM writes a question, reference answer and evidence span per chunk. Also ask for multi-hop questions across two chunks, and unanswerable ones.
2. **Human review is mandatory:** unreviewed synthetic questions copy the chunk's wording and are too easy. Rewrite them in user language.
3. **Cover hard cases:** unanswerable, multi-hop, tables, figures, identifiers, ambiguous, follow-ups, other languages.
4. **Label by document + page span**, not chunk ID.
5. **Split:** a dev set (tune on it) and a held-out test set (report on it). Otherwise you overfit your prompt to your eval.
6. **Production feedback loop:** every thumbs-down or failure becomes a new golden question.

Tools: RAGAS `TestsetGenerator`, DeepEval synthesizer, or your own script (you learn more writing your own).

### Step 6: Build the eval harness

```
eval/
├── golden.jsonl              # dev + test splits
├── run_eval.py               # --config configs/xyz.yaml --split dev
├── metrics/retrieval.py      # hit@k, recall@k, MRR, nDCG
├── metrics/generation.py     # judges: faithfulness, relevance, correctness, refusal
├── reports/2026-11-20_rerank50.md
└── results.md                # the ablation table (your interview story)
```

Requirements:
- **Config-driven:** chunker, embedding model, k, reranker, prompt version and model all come from config, so one command compares two configs.
- **Cached:** cache retrieval and LLM outputs by (config hash, question) so re-runs are cheap.
- **Per-category breakdown:** scores by question `type` (a bad table score hides inside a good average).
- **Latency and token counts per query** reported alongside quality (locally) plus an **estimated cloud cost** from those token counts, so you can predict the Bedrock bill before Phase 09.
- **Record the model IDs** (generator, embedding, reranker, judge) in every report row.
- **Failure dump:** for every miss, write the question, expected page, retrieved pages and answer to the report.

Then try **RAGAS** (or DeepEval, TruLens) on the same data and compare its numbers with your own metrics. Both can use a local OpenAI-compatible endpoint: point their LLM and embedding settings at `http://localhost:1234/v1` with your judge and embedding model names. Local judges may fail on these frameworks' long prompts more often than frontier models do. Log the parse failures and count them.

### Step 7: CI regression gate

```yaml
# .github/workflows/rag-eval.yml (excerpt)
- name: RAG regression gate
  run: python eval/run_eval.py --split test --min-hit-at-5 0.85 --min-mrr 0.70 --min-faithfulness 0.90
```

Fail the build when quality drops. Run the cheap retrieval metrics on every PR, and the full LLM-judged suite nightly or before release.

### Step 8: Observability and tracing

Log a **trace** for every request with spans for each step:

| Field | Why |
|-------|-----|
| query, rewritten query, route, filters | Debug query understanding |
| retrieved chunk IDs + scores (dense, BM25, RRF) | Debug retrieval |
| reranked order + scores | Debug ranking |
| final prompt token count, model ID, prompt version | Cost and reproducibility |
| answer, citations, refusal flag, grounding result | Quality |
| latency per step, total cost | Performance |
| user feedback (thumbs) | Online quality signal |

Tools: **Arize Phoenix** or **Langfuse** (both can run locally and are OpenTelemetry-based), LangSmith, or plain OpenTelemetry → Jaeger. Instrument `pipeline.py` with spans: `transform`, `retrieve.dense`, `retrieve.bm25`, `fuse`, `rerank`, `assemble`, `generate`.

**Online evaluation:**
- Sample 1–5% of production traces → LLM-judge faithfulness and relevance → dashboard
- Monitor no-answer rate, thumbs-down rate, p50/p95 latency, cost per query, and retrieval-score drift
- Alert on sudden changes (a broken ingestion job shows up as a spike in the no-answer rate)

### Step 9: Debugging playbook

| Retrieval | Answer | Most likely fix |
|-----------|--------|-----------------|
| Right chunk missing | Wrong | Parsing, chunking, embeddings, hybrid, query rewrite |
| Right chunk ranked low | Wrong / incomplete | Reranker, larger candidate k |
| Good | Unfaithful (adds facts) | Stricter prompt, fewer noisy chunks, grounding check, stronger model |
| Good | Faithful but incomplete | Larger k, parent expansion, bigger chunks |
| Good | Correct but slow or costly | Fewer chunks, smaller model, caching |

---

## Milestone

- `eval/` harness: config-driven, cached, retrieval + generation metrics, per-category breakdown, cost and latency
- Golden set of **100+** questions with dev/test split, reviewed by you
- Judge calibrated against 30+ of your own labels (agreement number recorded)
- CI workflow that fails on regression
- Phoenix or Langfuse tracing on every pipeline step; screenshot of a trace in your notes
- **Full ablation table** for every technique from Phases 01–05 on the *test* split

## Exit checklist

- [ ] I can define recall@k, precision@k, MRR and nDCG and compute them by hand.
- [ ] I can explain faithfulness vs answer relevance vs correctness.
- [ ] I know the biases of LLM judges and how to calibrate them.
- [ ] I can generate a synthetic test set and explain why human review matters.
- [ ] I can debug a wrong answer stage by stage from a trace.
- [ ] My CI blocks a change that lowers quality.

## Common pitfalls

- Only measuring end-to-end answer quality (you can't tell which stage broke).
- Tuning on the same questions you report on.
- Labelling by chunk ID, which breaks when chunking changes.
- Trusting an uncalibrated judge.
- Averages hiding category failures (tables at 30% inside an 85% overall score).

## Interview questions

1. How do you evaluate a RAG system?
2. What are recall@k and MRR? What does each tell you to fix?
3. How do you build a golden dataset?
4. Can you trust LLM-as-a-judge?
5. Users say answers are wrong. How do you debug?
6. What metric would you show a business stakeholder?

## Resources

- Es et al., 2023: *RAGAS* paper; docs.ragas.io
- TruLens *RAG Triad*; DeepEval docs
- Zheng et al., 2023: *Judging LLM-as-a-Judge with MT-Bench* (judge biases)
- Arize Phoenix and Langfuse docs
- Hamel Husain: *Your AI Product Needs Evals* (blog)
- Guide: Part 2 §9, §10; Interview Q22–Q26, Q30

## Next

→ [Phase 07: Advanced Paradigms](../07-advanced-paradigms/README.md)
