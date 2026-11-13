# Phase 04: Query Transformation & Pre-Retrieval Optimization

**Goal:** Bridge the vocabulary mismatch between how users ask and how documents are written, and send each query to the right place.
**Time:** about 1 week part-time.
**Guide sections:** Part 2 §8 (Production query pipeline, steps 3–4).

**Models this phase** (see [MODELS.md](../MODELS.md)):

| Role | Model | Why here |
|------|-------|----------|
| Rewrite, multi-query, HyDE, step-back, routing, self-query | **Qwen3-4B-Instruct-2507** via `helper()` | These calls sit *in front of* every search, so they must be fast; small models handle short rewriting tasks well |
| Comparison | Qwen3-30B-A3B-Instruct-2507 | Run the same transforms with the big model: is the quality gain worth the extra latency? |

---

## Why this phase matters

Users write "what about contractors?" (a follow-up with no subject), "my laptop won't charge lol" (informal), or "compare 2024 vs 2026 travel policy" (two questions in one). Documents say "Section 4.2: Leave Entitlements for Non-Permanent Staff". Even perfect chunks and a perfect index fail if the *query* is bad. These techniques fix the query **before** retrieval.

> **Engineer's caution:** every technique here adds an LLM call (200–600 ms plus cost). Use each only where your golden set shows it helps, and skip it when it doesn't.

---

## Learning steps (in order)

### Step 1: Conversational query rewriting (do this first; highest value)

Turn a follow-up into a **standalone** question using chat history.

```
History: "What is the sick-leave policy?"  →  "Employees get 12 days..."
User:    "what about contractors?"
Rewrite: "What is the sick-leave policy for contractors?"
```

```python
# rag/query.py
from rag.llm import helper

def rewrite(question: str, history: list[dict]) -> str:
    if not history:
        return question                       # skip the LLM call for first-turn questions
    convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-6:])
    return helper(
        f"<history>\n{convo}\n</history>\n<question>{question}</question>\n"
        "Rewrite the question as a standalone search query that keeps every specific "
        "name, number and code. Reply with the query only.",
        max_tokens=200).strip()
```

### Step 2: Multi-query retrieval (query expansion)

Generate 3–5 differently worded variants, retrieve for each, and fuse with RRF (you built it in Phase 03).
- Helps with recall when wording varies a lot.
- Costs: N retrievals + 1 LLM call; adds noise if the variants drift off topic.

```python
def multi_query(question: str, n: int = 4) -> list[str]:
    out = helper(f"Write {n} different search queries that would find documents "
                   f"answering: {question}\nOne per line, no numbering.", max_tokens=400)
    return [question] + [q.strip() for q in out.splitlines() if q.strip()][:n]
```

### Step 3: Query decomposition

Split a multi-part or multi-hop question into sub-questions, retrieve for each, and answer with all the evidence.
- "Compare the 2024 and 2026 travel policies on per diem" → ["2024 travel policy per diem", "2026 travel policy per diem"].
- *Least-to-most*: answer sub-questions sequentially when later ones depend on earlier answers. This leads into agentic RAG (Phase 07).

### Step 4: HyDE (Hypothetical Document Embeddings)

1. Ask the LLM to write a *hypothetical* answer passage (it may be factually wrong; that's fine).
2. Embed that passage instead of (or as well as) the question.
3. Search: a fake answer looks more like real answer passages than a short question does.

- Helps when questions and documents are worded very differently (short question, long technical docs).
- Risks: the hypothetical answer brings in the model's bias; adds latency. It can *hurt* on identifier lookups.

```python
def hyde_vector(question: str):
    fake = helper(f"Write a short passage from an internal document that answers: "
                    f"{question}", max_tokens=300)
    return embed_query(fake)   # or average with embed_query(question)
```

### Step 5: Step-back prompting

Generate a broader, more abstract question, then retrieve for **both** the original and the step-back question.
- "Can I carry over 7 unused sick days from 2025 to 2026?" → step-back: "What are the rules for carrying over unused leave?"
- Retrieves the underlying principle alongside the specific fact.

### Step 6: Self-query / metadata extraction

An LLM extracts structured filters from natural language:
- "Show me finance policies updated after March 2026" → `query="policies", filters={"department": "finance", "doc_date": {"gte": "2026-03-01"}}`
- Use **structured outputs** (a JSON schema) so the result is always valid, and validate allowed fields and values in code. LM Studio supports this through `response_format`:

```python
import json, os
from openai import OpenAI
client = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")

SCHEMA = {"name": "search_plan", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["query", "department", "date_from"],
    "properties": {
        "query": {"type": "string"},
        "department": {"type": ["string", "null"], "enum": ["HR", "Finance", "IT", None]},
        "date_from": {"type": ["string", "null"], "description": "YYYY-MM-DD"}}}}

r = client.chat.completions.create(
    model=os.getenv("HELPER_MODEL"), temperature=0,
    messages=[{"role": "user", "content": "Extract a search plan: finance policies updated after March 2026"}],
    response_format={"type": "json_schema", "json_schema": SCHEMA})
plan = json.loads(r.choices[0].message.content)
```
- **Never** let the LLM produce *security* filters (tenant, groups). Those come from the authenticated user, in code.

### Step 7: Query routing

Classify intent and route to the right source or strategy:

| Intent | Route |
|--------|-------|
| Greeting / chit-chat | No retrieval; answer directly |
| Policy question | Vector/hybrid search over the policy index |
| "How many tickets were opened last week?" | Text-to-SQL over the analytics database |
| Breaking news | Web search tool |
| Exact code lookup "HX-4410" | Keyword-heavy search (raise BM25 weight) |

Implementation options, cheapest first:
1. Rules / regex (an ID pattern triggers keyword search)
2. Embedding similarity to route descriptions ("semantic router")
3. A small classifier
4. An LLM with structured output (most flexible, slowest)

Routing is the first step toward agentic RAG, where the model chooses tools itself.

### Step 8: Put it together and measure

Build one entry point:

```python
def transform(question, history) -> QueryPlan:
    # returns: route, standalone query, variants, filters, use_hyde flag
    ...
```

Ablation (one technique at a time):

```
| Config                         | hit@5 | MRR  | p95 latency | LLM calls/query |
|--------------------------------|-------|------|-------------|-----------------|
| 03 hybrid baseline             |       |      |             | 0               |
| + rewrite (follow-ups only)    |       |      |             | 0–1             |
| + multi-query (4)              |       |      |             | 1               |
| + HyDE                         |       |      |             | 1               |
| + step-back                    |       |      |             | 1               |
| rewrite with 30B instead of 4B |       |      |             | 0–1             |
```

Add 10 **conversational follow-up** questions (with history) and 5 **multi-part** questions to the golden set; otherwise you can't measure rewriting and decomposition.

---

## Milestone

- `rag/query.py`: rewrite, multi-query, decomposition, HyDE, step-back, self-query (with structured output) and a router, all behind `transform()`
- Golden set grown to **55+** questions, including follow-ups with history
- Ablation table and a written decision: which transforms are on by default, which are conditional, and which you dropped

## Exit checklist

- [ ] I can explain the vocabulary mismatch problem.
- [ ] I can explain HyDE and when it hurts.
- [ ] I know the difference between multi-query, decomposition and step-back.
- [ ] My router never lets the LLM set security filters.
- [ ] I know the latency and cost each transform adds, and I skip transforms that don't pay off.

## Common pitfalls

- Rewriting every query, even first-turn ones that are already fine (wasted latency).
- Rewriter dropping the exact product code or number from the query.
- Using HyDE for identifier lookups.
- No follow-up questions in the golden set, so improvements are invisible.

## Interview questions

1. A user asks "what about contractors?" in a chat. What does your system do?
2. Explain HyDE. When would you not use it?
3. How do you route a query between a vector store, SQL and web search?
4. How do you keep query transformation from blowing your latency budget?

## Resources

- Gao et al., 2022: *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Zheng et al., 2023: *Take a Step Back* (step-back prompting)
- Ma et al., 2023: *Query Rewriting for Retrieval-Augmented LLMs*
- Zhou et al., 2022: *Least-to-Most Prompting*
- Guide: Part 2 §8; Interview Q26

## Next

→ [Phase 05: Post-Retrieval Processing & Reranking](../05-post-retrieval-reranking/README.md)
