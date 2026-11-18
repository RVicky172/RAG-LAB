# Doc 1: RAG and its evaluation

**Time:** ~2.5 hours  **You build:** a question-answering system over the 4 PDFs in `data/raw/`, plus a script that scores it.

---

## 1. The idea in one picture

An LLM doesn't know your documents. RAG (Retrieval-Augmented Generation) finds the relevant pieces first and pastes them into the prompt:

```
INGEST (once)                          QUERY (every question)
PDFs ─► text ─► chunks ─► vectors ─┐   question ─► vector ─► Qdrant: 5 closest chunks
                                   ▼                              │
                               Qdrant (vector DB)                 ▼
                                                    prompt = rules + chunks + question
                                                                  │
                                                                  ▼
                                                     LLM ─► answer with [source p.N]
```

| Word | Plain meaning |
|---|---|
| **Chunk** | A ~1,000-character piece of a page. Small enough to be specific, big enough to make sense. |
| **Embedding** | A list of 384 numbers that captures a text's meaning. Similar meaning → nearby numbers. |
| **Vector database** | Stores chunks + their embeddings and finds the closest ones to a question fast. We use **Qdrant**. |
| **Retrieval** | Step 1: find chunks. **Generation**: step 2: the LLM writes the answer from them. |

**Why these tools?** Qdrant is a real, widely used vector DB with a web dashboard. Its Python client embeds text for you (with the small, free `bge-small-en-v1.5` model on your CPU), so there is no separate embedding service. LM Studio runs the LLM (`gpt-oss-20b`) on your GPU for free.

---

## 2. Start the infrastructure (10 min)

```bash
# 1. Vector DB (Docker Desktop must be running)
docker compose up -d
# open http://localhost:6333/dashboard  -> empty for now

# 2. LLM
lms server start
lms load openai/gpt-oss-20b --context-length 32768 -y
lms ps                         # should list openai/gpt-oss-20b

# 3. Python packages (already in pyproject.toml)
uv sync
cp .env.example .env           # skip if .env exists
```

---

## 3. Settings and the LLM (10 min)

All code lives in `app/`. Every setting comes from `.env`, so moving to AWS later means editing `.env`, not code.

**`app/config.py`**
```python
"""All settings in one place, read from .env (local) or environment variables (AWS)."""
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "papers")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
```

**`app/llm.py`**
```python
"""One function returns the chat model. Local and AWS differ only in .env values:
LM Studio and Amazon Bedrock both speak the OpenAI chat API."""
from langchain.chat_models import init_chat_model
from app import config

def get_llm():
    return init_chat_model(config.LLM_MODEL, model_provider="openai", base_url=config.LLM_BASE_URL,
                           api_key=config.LLM_API_KEY, temperature=0)
```

`temperature=0` makes answers as repeatable as possible, which you want when you measure.

---

## 4. Ingest: PDFs → Qdrant (30 min)

**`app/ingest.py`**
```python
"""Read PDFs -> split into chunks -> store in Qdrant (embedding happens inside qdrant-client)."""
from pathlib import Path
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from app import config

def load_pages(folder="data/raw"):
    for pdf in sorted(Path(folder).glob("*.pdf")):
        for page_no, page in enumerate(pymupdf.open(pdf), start=1):
            yield {"source": pdf.name, "page": page_no, "text": page.get_text()}

def main():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = [{**p, "text": t} for p in load_pages() for t in splitter.split_text(p["text"])]
    print(f"{len(chunks)} chunks")

    client = QdrantClient(config.QDRANT_URL)
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)          # fresh start on every ingest
    client.create_collection(config.COLLECTION, vectors_config=models.VectorParams(
        size=client.get_embedding_size(config.EMBED_MODEL), distance=models.Distance.COSINE))
    client.upload_points(config.COLLECTION, points=[
        models.PointStruct(id=i, payload=c,
                           vector=models.Document(text=c["text"], model=config.EMBED_MODEL))
        for i, c in enumerate(chunks)], batch_size=64)
    print("stored:", client.count(config.COLLECTION).count)

if __name__ == "__main__":
    main()
```

```bash
uv run python -m app.ingest      # ~20 s; first run downloads the embedding model (~130 MB)
# 257 chunks / stored: 257
```

**What just happened:**
- The splitter cuts each page at paragraph, then line, then sentence boundaries, keeping chunks ≤ 1,000 characters. The 150-character **overlap** means a sentence cut at a boundary still appears whole in one chunk.
- `models.Document(text=..., model=...)` tells qdrant-client to embed the text itself.
- The **payload** (source, page, text) is stored next to each vector. That's how answers can cite a page.

**Look at it:** open http://localhost:6333/dashboard → `papers` → click a few points. Each one is a chunk with its payload.

---

## 5. Query: retrieve → augment → generate (30 min)

**`app/rag.py`**
```python
"""Retrieve -> augment -> generate."""
import sys
from qdrant_client import QdrantClient, models
from app import config
from app.llm import get_llm

client = QdrantClient(config.QDRANT_URL)

def search(question: str, k: int = 5) -> list[dict]:
    hits = client.query_points(config.COLLECTION, limit=k,
        query=models.Document(text=question, model=config.EMBED_MODEL)).points
    return [{**h.payload, "score": round(h.score, 3)} for h in hits]

PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".
Cite sources like [1706.03762v7.pdf p.5].

Context:
{context}

Question: {question}"""

def answer(question: str, k: int = 5) -> dict:
    chunks = search(question, k)
    context = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)
    reply = get_llm().invoke(PROMPT.format(context=context, question=question))
    return {"answer": reply.content, "contexts": chunks}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = answer(" ".join(sys.argv[1:]) or "How many attention heads does the base Transformer use?")
    print(result["answer"])
    print("\nSources:", [(c["source"], c["page"], c["score"]) for c in result["contexts"]])
```

```bash
uv run python -m app.rag
uv run python -m app.rag "What is the difference between RAG-Sequence and RAG-Token?"
uv run python -m app.rag "What is the capital of France?"      # should say "I don't know"
```

**Why the prompt says "ONLY the context" and "I don't know":** without it, the LLM fills gaps from memory, and you can't tell a grounded answer from a guess. The citation rule lets a user check the page.

---

## 6. Evaluation (45 min)

"It looked right on 3 questions" isn't evidence. You need a fixed question set and numbers you can compare before and after every change.

**The test set** is already in `eval/rag_questions.jsonl`: 11 questions, each with a `reference` answer and the expected `source` paper. One question (GPT-4) has no answer in the papers, to check that the system says "I don't know".

**Four metrics.** Each one points at a different part of the pipeline:

| Metric | Question it answers | Low score means fix… |
|---|---|---|
| `retrieval_hit` | Did the expected paper show up in the top 5? (plain Python, no LLM) | retrieval |
| `context_recall` | Do the retrieved chunks contain the facts in the reference? | retrieval (chunking, k) |
| `faithfulness` | Is every claim in the answer supported by the chunks? (hallucination check) | prompt / LLM |
| `answer_accuracy` | Does the answer match the reference? | anything; look at the other three |

The last three use **RAGAS**, a standard library that uses an LLM as a **judge**. Here the judge is the same local model.

**`eval/rag_eval.py`**
```python
"""Evaluate the RAG pipeline: one retrieval check + three RAGAS LLM-judge metrics.

Run: uv run python -m eval.rag_eval
"""
import asyncio, json, sys, time
from pathlib import Path
import instructor
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy, ContextRecall, Faithfulness
from app import config
from app.rag import answer

def judge_llm():
    """The judge: same model, through the OpenAI-compatible API."""
    client = AsyncOpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    # LM Studio accepts structured output only as a JSON schema, so tell RAGAS to use that mode
    return llm_factory(config.LLM_MODEL, client=client, mode=instructor.Mode.JSON_SCHEMA,
                       max_tokens=8192, reasoning_effort="low")   # gpt-oss thinks first: give it room

async def score(row, metrics):
    contexts = [c["text"] for c in row["contexts"]]
    recall, faith, acc = await asyncio.gather(
        metrics["recall"].ascore(user_input=row["question"], retrieved_contexts=contexts, reference=row["reference"]),
        metrics["faith"].ascore(user_input=row["question"], response=row["answer"], retrieved_contexts=contexts),
        metrics["acc"].ascore(user_input=row["question"], response=row["answer"], reference=row["reference"]))
    return {"context_recall": recall.value, "faithfulness": faith.value, "answer_accuracy": acc.value}

async def main():
    sys.stdout.reconfigure(encoding="utf-8")
    llm = judge_llm()
    metrics = {"recall": ContextRecall(llm=llm), "faith": Faithfulness(llm=llm), "acc": AnswerAccuracy(llm=llm)}
    rows = [json.loads(l) for l in Path("eval/rag_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    results = []
    for row in rows:
        t = time.perf_counter()
        out = answer(row["question"])
        row |= {"answer": out["answer"], "contexts": out["contexts"], "latency_s": round(time.perf_counter() - t, 1)}
        row["retrieval_hit"] = float(row["source"] is None or row["source"] in {c["source"] for c in out["contexts"]})
        row |= await score(row, metrics)
        results.append(row)
        print(f"{row['question'][:55]:55s} hit={row['retrieval_hit']:.0f} recall={row['context_recall']:.2f} "
              f"faith={row['faithfulness']:.2f} acc={row['answer_accuracy']:.2f} {row['latency_s']}s")

    keys = ["retrieval_hit", "context_recall", "faithfulness", "answer_accuracy", "latency_s"]
    print("\nMEAN  " + "  ".join(f"{k}={sum(r[k] for r in results) / len(results):.2f}" for k in keys))
    Path("eval/results").mkdir(exist_ok=True)
    Path("eval/results/rag_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
uv run python -m eval.rag_eval        # ~2.5 min; you may see library warnings: ignore them
```

When this was tested, the result was about `retrieval_hit=1.00 context_recall=0.91 faithfulness=0.91 answer_accuracy=0.91`. Write your numbers in the table in [README.md](../README.md).

### Read the failures, not just the mean

Open `eval/results/rag_eval.json` and look at every row scoring below 1. In the test run there were two:
- **"Which generator model does RAG use?"** The answer said "BART", but the reference says "BART-large", and none of the 5 retrieved chunks contained "large". That's a **retrieval** miss (`context_recall=0`), not an LLM mistake.
- **"How many attention heads…"** The answer "eight (8) heads [p.5]" was correct, yet `faithfulness=0`. That's a **judge error**. LLM judges are noisy, so a single low score is a reason to look, not a verdict.

---

## 7. Experiment: change one thing (30 min)

This is how improvement works in real teams: **change one setting, re-run the eval, keep the change only if the numbers improve.** Try one or two:

| Change | Where | Expectation |
|---|---|---|
| `k=5` → `k=8` | `answer(..., k=8)` in `rag.py` | higher context_recall, slower, more noise for the LLM |
| `chunk_size=1000` → `500` | `ingest.py`, then re-run ingest | more precise chunks, less context in each |
| Remove the "I don't know" line from the prompt | `rag.py` | watch the GPT-4 question start hallucinating |

Record each run as a row in the README results table.

---

## Checklist

- [ ] I can explain ingest vs query, and what a chunk, an embedding and a vector DB are.
- [ ] I saw my chunks in the Qdrant dashboard.
- [ ] I can say which metric points to retrieval and which to generation.
- [ ] I read at least two failing rows and decided whether retrieval, the LLM or the judge was wrong.
- [ ] I ran one experiment and recorded the before/after numbers.

**Next:** [Doc 2: AI agent and its evaluation](02-ai-agent-and-evaluation.md)
