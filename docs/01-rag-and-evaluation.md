# Doc 1: RAG and its evaluation

**Time:** ~3 hours  **You build:** a question-answering system over the 4 PDFs in `data/raw/`, plus an eval that scores it and writes an HTML report.

**Files you create in this doc:**

| Folder | Files | Step |
|---|---|---|
| `app/shared/` | `config.py`, `llm.py`, `tracing.py` | 3 |
| `app/rag/` | `ingest.py`, `query.py` | 4, 5 |
| `eval/shared/` | `judge.py`, `questions.py`, `status.py`, `html_page.py`, `style.css` | 6.4 |
| `eval/rag/` | `tokens.py`, `diagnose.py`, `report.py`, `evaluate.py` | 6.5 |
| `eval/` | `rebuild_reports.py` | 6.5 |

Plus an `__init__.py` in every folder. The question file `eval/rag/questions.jsonl` is already there.

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

**Why these tools?** Qdrant is a real, widely used vector DB with a web dashboard. Its Python client embeds text for you (with the small, free `bge-small-en-v1.5` model on your CPU), so there is no separate embedding service. LM Studio runs the LLM (`gpt-oss-20b`) on your GPU for free. **Arize Phoenix** records every step (search, LLM call, tokens, timing) so you can see what happened; it runs locally in Docker, so nothing leaves your machine.

---

## 2. Start the infrastructure (10 min)

```bash
# 1. Vector DB + tracing (Docker Desktop must be running)
docker compose up -d
# open http://localhost:6333/dashboard  -> Qdrant, empty for now
# open http://localhost:6006            -> Phoenix, empty for now

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

All app code lives in `app/`. Every setting comes from `.env`, so moving to AWS later means editing `.env`, not code.

By the end of Doc 4, `app/` looks like this. Code used by more than one system lives in `shared/`; every system has its own folder:

```
app/
├── shared/                 used by more than one system
│   ├── config.py           settings, read from .env                  (this step)
│   ├── llm.py              the chat model                            (this step)
│   ├── tracing.py          send every step to Phoenix                (this step)
│   ├── tools.py            search_papers + calculator                (Doc 2; both agents use them)
│   └── trace.py            print an agent's conversation             (Doc 2; both agents use it)
├── rag/                    Doc 1
│   ├── ingest.py           once: PDFs -> chunks -> Qdrant            python -m app.rag.ingest
│   └── query.py            per question: retrieve -> generate        python -m app.rag.query "..."
├── single_agent/           Doc 2
│   └── agent.py            one agent with both tools                 python -m app.single_agent.agent "..."
├── multi_agent/            Doc 3
│   ├── specialists.py      researcher + analyst, wrapped as tools
│   └── supervisor.py       the supervisor that delegates to them     python -m app.multi_agent.supervisor "..."
└── langgraph_agent/        Doc 4
    ├── nodes.py            the state and the steps of the graph
    └── graph.py            wires the steps into a LangGraph graph    python -m app.langgraph_agent.graph "..."
```

Every folder also has an `__init__.py` (shown below with each folder). It makes the folder a Python package, and its docstring says what's inside.

**`app/__init__.py`**
```python
"""
The paper assistant: four ways to answer questions about the 4 AI papers in data/raw/.

    shared/        used by every system: settings, the LLM, the tools, trace printing
    rag/           Doc 1: RAG (ingest the PDFs once, then query them)
    single_agent/  Doc 2: one agent that decides which tools to call
    multi_agent/   Doc 3: a supervisor that delegates to two specialist agents
    langgraph_agent/  Doc 4: a self-correcting RAG agent, built step by step with LangGraph
"""
```

`config.py`, `llm.py` and `tracing.py` go in `app/shared/`, because every system (RAG, single agent, multi-agent) uses them.

**`app/shared/__init__.py`**
```python
"""
Code shared by more than one system.

    config.py   settings, read from .env                    (every system)
    llm.py      the chat model                              (every system)
    tools.py    search_papers + calculator                  (single agent, multi-agent)
    trace.py    print an agent's conversation step by step  (single agent, multi-agent)
    tracing.py  send every step to Arize Phoenix (local)     (every system, every eval)
"""
```

`config.py` reads every setting from `.env`, with a default for local use:

**`app/shared/config.py`**
```python
"""
Settings for the whole project, kept in one place.

Every value is read from an environment variable. Locally these come from the
`.env` file. On AWS they come from the server's environment. If a variable is
not set, the default below is used, and the defaults match the local setup
(LM Studio + Qdrant in Docker).

Why do it this way? The code never changes between local and AWS. Only the
settings change.
"""

import os

from dotenv import load_dotenv

# Copy the values from `.env` into os.environ. Variables that are already set
# (for example on AWS) are NOT overwritten.
load_dotenv()

# --- Vector database (Qdrant) -------------------------------------------------
# Where Qdrant runs, and the name of the "table" (collection) holding our chunks.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "papers")

# --- Embedding model ----------------------------------------------------------
# Turns text into a vector (a list of numbers) so similar texts end up close
# together. Small, fast, and runs on the CPU through fastembed.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chat model (LLM) ---------------------------------------------------------
# LM Studio (local) and Bedrock (AWS) both speak the OpenAI API, so we only
# need a model name, a URL and a key.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")  # LM Studio accepts any key

# The most tokens one LLM call can hold (prompt + reply together). Locally it
# is what we loaded the model with (`lms load ... --context-length 32768`).
# On Bedrock gpt-oss-20b allows 128k, so set CONTEXT_WINDOW=131072 there.
# Only used to report how full the window is.
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "32768"))

# --- Tracing (Arize Phoenix) -----------------------------------------------------
# Phoenix runs locally in Docker (docker-compose.yml) and records every LLM call,
# tool call and agent step, with tokens and timing. Open PHOENIX_URL to see them.
# Set TRACING=false to run without it (e.g. when Phoenix isn't running).
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
TRACING = os.getenv("TRACING", "true").lower() == "true"
```

`llm.py` creates the chat model. LM Studio and Bedrock both speak the OpenAI API, so one function covers both:

**`app/shared/llm.py`**
```python
"""
Creates the chat model (LLM) that the rest of the project uses.

LM Studio (local) and Amazon Bedrock (AWS) both speak the OpenAI chat API, so
one function covers both. Moving to AWS only changes the values in `.env`.
"""

from langchain.chat_models import init_chat_model

from app.shared import config


def get_llm():
    """Return a LangChain chat model connected to our LLM server.

    Returns:
        A LangChain chat model. Call `.invoke(prompt)` on it and read the
        reply text from `.content`.
    """
    return init_chat_model(
        config.LLM_MODEL,
        model_provider="openai",       # use the OpenAI client (LM Studio and Bedrock both support it)
        base_url=config.LLM_BASE_URL,  # where the server runs: LM Studio or Bedrock
        api_key=config.LLM_API_KEY,
        temperature=0,                 # no randomness, so eval runs are easier to compare
    )
```

`temperature=0` makes answers as repeatable as possible, which you want when you measure.

**Tracing.** A **trace** is the record of one request: every step (search, LLM call, tool call), nested like a tree, with its input, output, tokens and time. Companies use tools like LangSmith or Langfuse for this; we use **Arize Phoenix**, which runs locally in Docker (started in step 2) and needs no account. We don't write tracing code: "instrumentors" hook into LangChain and the OpenAI client and record every call. `start_tracing()` turns that on; each command calls it once at the start.

**`app/shared/tracing.py`**
```python
"""
Send traces to Arize Phoenix, a tracing tool that runs locally (see docker-compose.yml).

A trace is the record of one request: every LLM call, tool call and agent
step, nested like a tree, with its input, output, tokens and time. Phoenix
shows them at http://localhost:6006. Nothing leaves your machine.

We don't write any tracing code ourselves. "Instrumentors" from the
OpenInference project hook into the libraries and record every call:
    LangChain   every chain, agent, tool and LLM call of the app
    OpenAI      the RAGAS judge's calls (RAGAS uses the OpenAI client directly, not LangChain)

Call start_tracing() once, at the start of a command.
"""

import openai
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from phoenix.otel import register

from app.shared import config


def start_tracing(project: str, judge: bool = False):
    """Turn on tracing for this run.

    Args:
        project: The Phoenix project the traces go to, e.g. "rag" or "rag-eval".
            Each project is a separate list in the Phoenix UI.
        judge: Also record the RAGAS judge's calls (for the evals).

    Returns:
        A tracer. The evals use it to group everything one question does
        (answer, tools, judge) under one trace. When TRACING is off it is a
        do-nothing tracer, so the calling code doesn't change.
    """
    if not config.TRACING:
        return trace.get_tracer(project)

    # Where to send the traces, and which project they belong to.
    # batch=True: send them in the background, so tracing doesn't slow the app down.
    provider = register(
        endpoint=f"{config.PHOENIX_URL}/v1/traces",
        project_name=project,
        batch=True,
        verbose=False,
    )

    # Record every LangChain step: agents, tools, and LLM calls.
    LangChainInstrumentor().instrument(tracer_provider=provider)

    if judge:
        # The judge calls the LLM through the *async* OpenAI client, our app
        # through LangChain, which uses the *sync* OpenAI client underneath.
        # The OpenAI instrumentor hooks both clients, so we put the sync one
        # back: LangChain already records those calls, and recording them
        # twice would double every token count in Phoenix.
        sync_request = openai.OpenAI.request
        OpenAIInstrumentor().instrument(tracer_provider=provider)
        openai.OpenAI.request = sync_request

    return provider.get_tracer(project)


def trace_url(trace_id: str) -> str:
    """The Phoenix page of one trace, e.g. for a link in the HTML report."""
    return f"{config.PHOENIX_URL}/redirects/traces/{trace_id}"


def current_trace_id() -> str:
    """The id of the trace we're in right now, as Phoenix shows it (32 hex characters)."""
    return format(trace.get_current_span().get_span_context().trace_id, "032x")
```


---

## 4. Ingest: PDFs → Qdrant (30 min)

**`app/rag/__init__.py`**
```python
"""
RAG (Doc 1): answer questions from the papers in two stages.

    ingest.py   once:          PDFs -> chunks -> vectors in Qdrant   (python -m app.rag.ingest)
    query.py    per question:  retrieve -> augment -> generate       (python -m app.rag.query "...")
"""
```

`ingest.py` reads the PDFs, splits them into chunks and stores them in Qdrant. You run it once (and again after you change the chunking):

**`app/rag/ingest.py`**
```python
"""
Ingestion: load the PDFs into the vector database.

Steps:
    1. Read every page of every PDF in `data/raw/`.
    2. Split each page into small chunks of text.
    3. Store the chunks in Qdrant. qdrant-client turns each chunk into a vector
       (an "embedding") for us, so we never call the embedding model ourselves.

Why chunks? An LLM can only read a limited amount of text, and a small chunk
about one topic is easier to find with search than a whole page.

Run: uv run python -m app.rag.ingest
"""

from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from app.shared import config


def load_pages(folder: str = "data/raw"):
    """Read every page of every PDF in a folder.

    Args:
        folder: Folder that holds the PDF files.

    Yields:
        One dict per page: `{"source": file name, "page": page number
        (starting at 1), "text": the page's text}`. The source and page are
        kept so the answer can cite where it found something.
    """
    for pdf in sorted(Path(folder).glob("*.pdf")):
        for page_no, page in enumerate(pymupdf.open(pdf), start=1):
            yield {"source": pdf.name, "page": page_no, "text": page.get_text()}


def split_into_chunks(pages) -> list[dict]:
    """Split each page into small, overlapping chunks.

    Args:
        pages: The page dicts from `load_pages()`.

    Returns:
        A list of chunk dicts. Each one has the same keys as a page
        (source, page, text), but `text` holds only the chunk.
    """
    # The splitter tries to cut at paragraphs first, then lines, then words,
    # so a chunk rarely stops in the middle of a sentence.
    # chunk_size: the most characters in one chunk.
    # chunk_overlap: neighbouring chunks share this many characters, so a
    # sentence that sits on a cut still appears whole in one of them.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    chunks = []
    for page in pages:
        for text in splitter.split_text(page["text"]):
            chunks.append({**page, "text": text})  # copy source/page, replace the text
    return chunks


def main():
    """Rebuild the Qdrant collection from the PDFs."""
    chunks = split_into_chunks(load_pages())
    print(f"{len(chunks)} chunks")

    client = QdrantClient(config.QDRANT_URL)

    # Start fresh every time, so old chunks from an earlier run never mix in.
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)

    # A collection must know the vector size up front (384 for bge-small).
    # COSINE means "how similar are these two vectors", the usual choice for text.
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(config.EMBED_MODEL),
            distance=models.Distance.COSINE,
        ),
    )

    # Each chunk becomes one "point":
    #   id      -> a unique number
    #   payload -> the data we get back from a search (source, page, text)
    #   vector  -> models.Document asks qdrant-client to embed the text for us
    points = [
        models.PointStruct(
            id=i,
            payload=chunk,
            vector=models.Document(text=chunk["text"], model=config.EMBED_MODEL),
        )
        for i, chunk in enumerate(chunks)
    ]

    # batch_size: send 64 points per request instead of everything at once.
    # wait=True: wait until Qdrant has saved them, so the count below is correct.
    client.upload_points(config.COLLECTION, points=points, batch_size=64, wait=True)
    print("stored:", client.count(config.COLLECTION).count)


if __name__ == "__main__":
    main()
```

```bash
uv run python -m app.rag.ingest  # ~20 s; first run downloads the embedding model (~130 MB)
# 257 chunks / stored: 257
```

**What just happened:**
- The splitter cuts each page at paragraph, then line, then sentence boundaries, keeping chunks ≤ 1,000 characters. The 150-character **overlap** means a sentence cut at a boundary still appears whole in one chunk.
- `models.Document(text=..., model=...)` tells qdrant-client to embed the text itself.
- The **payload** (source, page, text) is stored next to each vector. That's how answers can cite a page.

**Look at it:** open http://localhost:6333/dashboard → `papers` → click a few points. Each one is a chunk with its payload.

---

## 5. Query: retrieve → augment → generate (30 min)

`query.py` answers one question: search Qdrant, put the chunks in the prompt, ask the LLM. It also returns the chunks and the token usage, because the eval needs both:

**`app/rag/query.py`**
```python
"""
RAG = Retrieval-Augmented Generation. It answers a question in three steps:

    1. Retrieve: find the chunks in Qdrant that are most similar to the question.
    2. Augment:  paste those chunks into the prompt as "context".
    3. Generate: the LLM writes an answer using only that context.

Why? The LLM doesn't know what's in our PDFs. Giving it the right pages lets
it answer from them and cite them, instead of guessing.

Run: uv run python -m app.rag.query "your question"
"""

import sys
import threading

from opentelemetry import trace
from qdrant_client import QdrantClient, models

from app.shared import config
from app.shared.llm import get_llm
from app.shared.tracing import start_tracing

# One shared Qdrant connection for the whole program. Reconnecting on every
# question would be slower.
client = QdrantClient(config.QDRANT_URL)

# The embedding model built into qdrant-client is not thread-safe. The eval
# script asks several questions in parallel threads, so we let only one search
# run at a time. A search takes about 10 ms, so this costs almost nothing.
# The slow part (the LLM call) still runs in parallel.
_search_lock = threading.Lock()

# Tracing: LangChain records the LLM call by itself, but the Qdrant search is our
# own code, so we record it ourselves. A "span" is one step in a trace. The
# attribute names are the ones Phoenix understands (OpenInference conventions):
# a RETRIEVER span with its documents is shown as a list of chunks with scores.
# When tracing is off, these spans do nothing.
tracer = trace.get_tracer(__name__)


def search(question: str, k: int = 5) -> list[dict]:
    """Retrieve: find the k chunks most similar to the question.

    Args:
        question: The user's question in plain text.
        k: How many chunks to return.

    Returns:
        A list of chunk dicts, best match first. Each one has
        `source`, `page`, `text`, and `score` (cosine similarity, higher is closer).
    """
    with tracer.start_as_current_span("search") as span, _search_lock:
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("input.value", question)

        # models.Document: qdrant-client embeds the question with the same
        # model used during ingest, then finds the nearest vectors.
        hits = client.query_points(
            config.COLLECTION,
            query=models.Document(text=question, model=config.EMBED_MODEL),
            limit=k,
        ).points
        chunks = [{**hit.payload, "score": round(hit.score, 3)} for hit in hits]

        # Record each chunk, so Phoenix can show what was retrieved.
        for i, c in enumerate(chunks):
            span.set_attribute(f"retrieval.documents.{i}.document.id", f"{c['source']} p.{c['page']}")
            span.set_attribute(f"retrieval.documents.{i}.document.score", c["score"])
            span.set_attribute(f"retrieval.documents.{i}.document.content", c["text"])

    return chunks


# The instructions for the LLM. {context} and {question} are filled in by answer().
# - "ONLY the context" keeps the model from making things up (hallucinating).
# - "I don't know" gives it a safe answer when the search found nothing useful.
# - The citation format lets us check where each fact came from.
PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".
Cite sources like [1706.03762v7.pdf p.5].

Context:
{context}

Question: {question}"""


def build_context(chunks: list[dict]) -> str:
    """Augment: turn the chunks into one text block for the prompt.

    Each chunk gets a `[file p.N]` label so the LLM can cite it.

    Args:
        chunks: Chunk dicts from `search()`.

    Returns:
        The chunks joined into one string, separated by blank lines.
    """
    return "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)


def answer(question: str, k: int = 5) -> dict:
    """Run the full RAG pipeline: retrieve, augment, generate.

    Args:
        question: The user's question.
        k: How many chunks to give the LLM as context.

    Returns:
        `{"answer": the LLM's reply, "contexts": the chunks it was given,
        "usage": token counts}`. The contexts and usage are returned as well
        because the eval scores and reports them. `usage` is what the server
        reported, e.g. `{"input_tokens": 1500, "output_tokens": 200,
        "output_token_details": {"reasoning": 120}, ...}`.
    """
    # One span around the whole pipeline, so search + LLM call show up as one trace.
    with tracer.start_as_current_span("rag") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", question)

        chunks = search(question, k)                                       # 1. retrieve
        prompt = PROMPT.format(context=build_context(chunks), question=question)  # 2. augment
        reply = get_llm().invoke(prompt)                                   # 3. generate

        span.set_attribute("output.value", reply.content)
    return {"answer": reply.content, "contexts": chunks, "usage": reply.usage_metadata or {}}


if __name__ == "__main__":
    # The Windows console uses cp1252 by default and fails on some characters
    # in the papers, so switch it to UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    start_tracing("rag")  # every run shows up in Phoenix, project "rag"

    # Use the command-line words as the question, or a default example.
    question = " ".join(sys.argv[1:]) or "How many attention heads does the base Transformer use?"
    result = answer(question)

    print(result["answer"])
    print("\nSources:", [(c["source"], c["page"], c["score"]) for c in result["contexts"]])
    print(f"Trace (tokens, timing): {config.PHOENIX_URL} (project rag)")
```

```bash
uv run python -m app.rag.query
uv run python -m app.rag.query "What is the difference between RAG-Sequence and RAG-Token?"
uv run python -m app.rag.query "What is the capital of France?"      # should say "I don't know"
```

**Why the prompt says "ONLY the context" and "I don't know":** without it, the LLM fills gaps from memory, and you can't tell a grounded answer from a guess. The citation rule lets a user check the page.

**Look at the trace:** open http://localhost:6006 → project `rag` → click the newest trace. You see the `rag` step with two steps inside: `search` (the 5 chunks with their scores) and `ChatOpenAI` (the full prompt, the answer, and the tokens). LangChain's LLM call is recorded automatically; the Qdrant search is our own code, so `query.py` records it with a small "span" (one step of a trace).

---

## 6. Evaluation (75 min)

"It looked right on 3 questions" isn't evidence. You need a fixed question set and numbers you can compare before and after every change.

### 6.1 What we measure

**The test set** is already in `eval/rag/questions.jsonl`: 11 questions, each with a `reference` answer and the expected `source` paper. One question (GPT-4) has no answer in the papers, to check that the system says "I don't know".

**Four scores**, each from 0 to 1. Each one points at a different part of the pipeline:

| Score | Question it answers | Low score means fix… |
|---|---|---|
| `retrieval_hit` | Did the expected paper show up in the top 5? (plain Python, no LLM) | retrieval |
| `context_recall` | Do the retrieved chunks contain the facts in the reference? | retrieval (chunking, k) |
| `faithfulness` | Is every claim in the answer supported by the chunks? (hallucination check) | prompt / LLM |
| `answer_accuracy` | Does the answer match the reference? | anything; look at the other three |

The last three use **RAGAS**, a standard library that uses an LLM as a **judge**: an LLM that grades another LLM's answer. Here the judge is the same local model.

### 6.2 Tokens

The eval also counts **tokens**, the pieces of text an LLM reads and writes. Tokens decide the cost (Bedrock bills per token), the speed, and whether a prompt fits the **context window** (the most tokens one call can hold: 32,768 here). For the answer call, the eval splits the input into its parts:

| Part | What it is |
|---|---|
| Instructions | The fixed rules in `PROMPT` |
| Context | The 5 retrieved chunks: usually ~85% of the input |
| Question | The user's question |
| Chat format | ~67 tokens the server wraps around every prompt (system header, role markers). You don't write them, but they count. |
| Output: reasoning | gpt-oss "thinks" before it answers. Generated and billed, but not shown. |
| Output: answer | The text the user sees |

**Phoenix shows the exact tokens of every call**, including the judge's: each eval question becomes one trace with the answer call and its ~5 judge calls, and the project page shows the total of the whole run. What Phoenix can't show is what the input is made of, so `eval/rag/tokens.py` counts each part with gpt-oss's tokenizer (`tiktoken`, already in `pyproject.toml`) and the report shows the split. The judge uses about 7× the tokens of the answer itself.

### 6.3 The eval folder

Like `app/`, `eval/` has one folder per system plus a `shared/` folder for what every eval uses. Each eval folder keeps its own questions and writes its own `results/`:

```
eval/
├── shared/                 used by every eval
│   ├── judge.py            the LLM judge
│   ├── questions.py        read a question file, save the results
│   ├── status.py           score thresholds and the pass / check / fail result
│   ├── html_page.py        building blocks for the HTML reports
│   └── style.css           how the reports look (copied into each page)
├── rag/                    Doc 1
│   ├── questions.jsonl     the test set (already there)
│   ├── tokens.py           split the answer call into instructions / context / question / output
│   ├── diagnose.py         pass / check / fail per question, and the likely cause
│   ├── report.py           write results/report.html
│   ├── evaluate.py         the eval you run                          python -m eval.rag.evaluate
│   └── results/            output: results.json + report.html (git-ignored)
├── single_agent/           Doc 2
├── multi_agent/            Doc 3
├── langgraph_agent/        Doc 4 (reuses the RAG eval)
└── rebuild_reports.py      rebuild every report from saved results   python -m eval.rebuild_reports
```

### 6.4 The shared eval files

These files are used by all three evals, so you write them once, here.

Every folder gets an `__init__.py` whose docstring says what's inside:

**`eval/__init__.py`**
```python
"""
The evals: one folder per system, plus the code they share.

    shared/        used by every eval: the judge, question files, the HTML report blocks
    rag/           Doc 1: evaluate the RAG pipeline       (python -m eval.rag.evaluate)
    single_agent/  Doc 2: evaluate the single agent       (python -m eval.single_agent.evaluate)
    multi_agent/   Doc 3: evaluate the multi-agent system (python -m eval.multi_agent.evaluate)
    langgraph_agent/  Doc 4: evaluate the LangGraph agent  (python -m eval.langgraph_agent.evaluate)
    rebuild_reports.py  rebuild every HTML report from the saved results (python -m eval.rebuild_reports)

Each eval folder holds its own questions and writes its own results/ folder:
results.json (every row) and report.html (open it in a browser). Every question is
also a trace in Phoenix (http://localhost:6006), in a project per eval.
"""
```

The shared folder:

**`eval/shared/__init__.py`**
```python
"""
Code shared by every eval (RAG, single agent, multi-agent).

    judge.py       the LLM judge
    questions.py   read a question file, save the results
    status.py      score thresholds and the pass / check / fail result
    html_page.py   building blocks for the HTML reports (page, cards, bars, question rows)
    style.css      the look of every report (copied into the HTML, so each page is one file)
"""
```

`judge.py` sets up the RAGAS judge. Its calls show up in Phoenix with the rest of each question:

**`eval/shared/judge.py`**
```python
"""
The LLM judge, used by every eval.

An "LLM judge" is an LLM that grades another LLM's work. RAGAS provides the
judge prompts; here the judge is the same model as the app. Its calls (and
their tokens) show up in Phoenix, under the eval's project.
"""

import instructor
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from app.shared import config


def judge_llm():
    """Create the LLM judge that RAGAS uses to score the answers.

    It is the same model as the app, called through the OpenAI-compatible API.

    Returns:
        A RAGAS LLM object to pass to each metric.
    """
    # RAGAS needs the async client so it can send several judge calls at once.
    client = AsyncOpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    return llm_factory(
        config.LLM_MODEL,
        client=client,
        # The judge must reply in a fixed JSON shape. LM Studio only accepts
        # that through a JSON schema (it rejects the default "json_object" mode).
        mode=instructor.Mode.JSON_SCHEMA,
        # gpt-oss "thinks" before it answers. With the default 1,024-token limit
        # it runs out of room mid-thought, so give it more and ask it to think less.
        max_tokens=8192,
        reasoning_effort="low",
    )
```

`questions.py` reads a question file and saves the results as JSON:

**`eval/shared/questions.py`**
```python
"""
Read a question file and save results: the same for every eval.
"""

import json
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    """Read a question file. JSONL means one JSON object per line.

    Args:
        path: Path to the .jsonl file.

    Returns:
        One dict per question, e.g. `{"question", "reference", ...}`.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]  # skip empty lines


def save_results(rows: list[dict], path: Path) -> None:
    """Save every result row as readable JSON, creating the folder if needed.

    Args:
        rows: The result rows (question, answer, scores, tokens, ...).
        path: Where to write the .json file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
```

`status.py` holds the thresholds every report uses: a score ≥ 0.8 is good (green), 0.5–0.8 needs a check (amber), below 0.5 is bad (red):

**`eval/shared/status.py`**
```python
"""
Grading shared by every eval: score thresholds, colours and the pass/check/fail result.

Each eval has its own rules for WHY a question failed (rag/diagnose.py,
single_agent/diagnose.py); this file holds what they have in common.
"""

from dataclasses import dataclass

# Score thresholds, used for every colour and status in the report.
GOOD = 0.8  # score >= GOOD          -> good
BAD = 0.5   # BAD <= score < GOOD    -> check;  score < BAD -> bad


@dataclass
class Diagnosis:
    """The result for one question.

    status: "pass", "check" or "fail".
    area:   where the problem probably is, e.g. "Retrieval" or "LLM". Empty when it passed.
    reason: one plain sentence explaining it. Empty when there is nothing to say.
    """

    status: str
    area: str = ""
    reason: str = ""


def level(score: float) -> str:
    """Grade one score: "good", "check" or "bad"."""
    if score >= GOOD:
        return "good"
    return "check" if score >= BAD else "bad"
```

The next two only build the report pages: plain HTML and CSS, nothing to learn about RAG. Copy them as they are.

<details><summary><b><code>eval/shared/html_page.py</code></b> (click to show)</summary>

**`eval/shared/html_page.py`**
```python
"""
The building blocks shared by every HTML report (RAG and agents).

    page()           the whole HTML document: title, styles, body
    stat()           one big-number card
    token_bar()      a bar split into coloured parts, plus legend() for its colour key
    trace_link()     a link to a question's trace in Phoenix, and phoenix_panel() for the project
    verdict()        the coloured "where is the problem" box of a question
    question_row()   one clickable question row (status, question, a few values) with details
    question_list()  a titled list of question rows with a column header

Each function returns a piece of HTML as a string. The look lives in style.css.
No libraries and no JavaScript, so each report is one file you can open anywhere.
"""

import html
from datetime import datetime
from pathlib import Path

from app.shared import config
from app.shared.tracing import trace_url
from eval.shared.status import Diagnosis

CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")


def esc(text) -> str:
    """Escape text for HTML, so a '<' in an answer can't break the page."""
    return html.escape(str(text))


def n(value: float) -> str:
    """A whole number with thousands separators: 12345 -> '12,345'."""
    return f"{value:,.0f}"


def stat(value: str, label: str, css: str = "") -> str:
    """One card with a big number and a short label under it."""
    return f'<div class="panel stat"><div class="value num {css}">{value}</div><div class="label">{label}</div></div>'


def token_bar(parts: list[tuple[str, str, float]]) -> str:
    """A bar split into coloured parts; each part's width is its share of the total.

    Args:
        parts: (name, colour, tokens) for each part, in display order.
    """
    total = sum(tokens for _, _, tokens in parts) or 1
    spans = "".join(f'<span style="width:{tokens / total:.2%};background:{color}" '
                    f'title="{name}: {n(tokens)} tokens"></span>'
                    for name, color, tokens in parts)
    return f'<div class="bar">{spans}</div>'


def legend(parts: list[tuple[str, str, float]]) -> str:
    """The colour key for a token bar (same parts list as token_bar)."""
    items = "".join(f'<span><i class="dot" style="background:{color}"></i>{name}</span>'
                    for name, color, _ in parts)
    return f'<div class="legend">{items}</div>'


def trace_link(row: dict, text: str = "Open trace ↗") -> str:
    """A link to the question's trace in Phoenix, or a dash if it has none (tracing was off)."""
    if not row.get("trace_id"):
        return '<span class="muted">–</span>'
    return f'<a href="{trace_url(row["trace_id"])}" target="_blank" rel="noopener">{text}</a>'


def phoenix_panel(project: str) -> str:
    """A box pointing to the Phoenix project of this eval, where every call's tokens and timing are."""
    return f"""
<div class="panel" style="margin-top:12px">
  <p><b>Every LLM call, tool call and judge call is traced in Phoenix</b>, with its tokens and timing:
     project <code>{esc(project)}</code> at <a href="{config.PHOENIX_URL}" target="_blank" rel="noopener">{config.PHOENIX_URL}</a>.
     Its project page shows the total tokens and latency of this run; each question below links to its own trace.</p>
</div>"""


def verdict(diag: Diagnosis) -> str:
    """The coloured box saying where the problem probably is. Empty when there's nothing to say."""
    if not diag.reason:
        return ""
    return f'<div class="verdict {diag.status}"><b>{esc(diag.area)}:</b> {esc(diag.reason)}</div>'


def question_row(num: int, question: str, diag: Diagnosis, cells: str, details: str) -> str:
    """One question: a row that opens to show its details. Problem questions start open.

    Args:
        num: Question number.
        question: The question text.
        diag: Its diagnosis (gives the status pill).
        cells: The value cells for the row, as <span>s (5 of them: they fill the 5 right-hand columns).
        details: The HTML shown when the row is open.
    """
    return f"""
<details class="q" id="q{num}" {"open" if diag.status != "pass" else ""}>
  <summary>
    <span><span class="pill {diag.status}">{diag.status.upper()}</span></span>
    <span class="muted">Q{num}</span>
    <span>{esc(question)}</span>
    {cells}
  </summary>
  <div class="qbody">{details}</div>
</details>"""


def question_list(title: str, note: str, columns: list[str], rows: list[str]) -> str:
    """A titled list of question rows under a column header. Returns "" when there are no rows.

    Args:
        title: Section title, e.g. "Needs attention".
        note: One muted line under the title.
        columns: Names of the 5 value columns, e.g. ["Hit", "Recall", "Faith", "Acc", "Tokens"].
        rows: The rows, from question_row().
    """
    if not rows:
        return ""
    header = ('<div class="qhead"><span>Status</span><span>#</span><span>Question</span>'
              + "".join(f"<span>{name}</span>" for name in columns) + "</div>")
    return (f'<h2>{title} ({len(rows)})</h2><p class="muted small">{note}</p>'
            f'<div class="qlist">{header}{"".join(rows)}</div>')


def page(title: str, subtitle: str, body: str) -> str:
    """The whole HTML document.

    Args:
        title: Page title, e.g. "RAG eval report".
        subtitle: What was run, e.g. "11 questions · 4 in parallel" (date and model are added).
        body: Everything below the title.
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <h1>{esc(title)}</h1>
  <p class="muted">{datetime.now():%d %b %Y, %H:%M} · {esc(config.LLM_MODEL)} · {subtitle}</p>
  {body}
</main>
</body>
</html>"""
```

</details>

<details><summary><b><code>eval/shared/style.css</code></b> (click to show)</summary>

**`eval/shared/style.css`**
```css
/* Look of every eval report. html_page.py copies this file into each page. */

/* --- Colours (light, then dark) ------------------------------------------- */
:root {
  --bg: #f7f7f8; --panel: #ffffff; --text: #1d1d1f; --muted: #6e6e73; --line: #e3e3e6;
  --good: #1a7f37; --check: #9a6700; --bad: #cf222e;
  --good-soft: #dafbe1; --check-soft: #fff8c5; --bad-soft: #ffebe9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --panel: #161b22; --text: #e6edf3; --muted: #8d96a0; --line: #30363d;
    --good: #3fb950; --check: #d29922; --bad: #f85149;
    --good-soft: #12261e; --check-soft: #272115; --bad-soft: #25171c;
  }
}

/* --- Page --------------------------------------------------------------------- */
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif;
}
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0; }
h2 { font-size: 18px; margin: 40px 0 12px; }
h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 20px 0 8px; }
p { margin: 0 0 8px; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.num { font-variant-numeric: tabular-nums; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 20px; }

/* --- Score and status colours ------------------------------------------------- */
.good { color: var(--good); }
.check { color: var(--check); }
.bad { color: var(--bad); }
.pill { display: inline-block; min-width: 54px; text-align: center; padding: 1px 8px; border-radius: 999px;
        font-size: 12px; font-weight: 600; }
.pill.pass { background: var(--good-soft); color: var(--good); }
.pill.check { background: var(--check-soft); color: var(--check); }
.pill.fail { background: var(--bad-soft); color: var(--bad); }

/* --- Summary at the top ----------------------------------------------------- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 20px; }
.stat .value { font-size: 30px; font-weight: 700; line-height: 1.2; }
.stat .label { font-size: 13px; color: var(--muted); }

table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
tr:last-child td { border-bottom: none; }
td.score { font-weight: 700; font-size: 17px; white-space: nowrap; }

/* --- Token bar ------------------------------------------------------------------ */
.bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; background: var(--line); margin: 8px 0; }
.bar span { height: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 13px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }

/* --- Question list: one row per question, click to open ----------------------- */
.qlist { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.qhead, .q > summary {
  display: grid; align-items: center; gap: 10px;
  grid-template-columns: 64px 36px 1fr repeat(4, 52px) 70px;
  padding: 10px 16px;
}
.qhead { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; border-bottom: 1px solid var(--line); }
.qhead span:nth-child(n+4), .q > summary span:nth-child(n+4) { text-align: right; }
.q { border-bottom: 1px solid var(--line); }
.q:last-child { border-bottom: none; }
.q > summary { cursor: pointer; list-style: none; }
.q > summary::-webkit-details-marker { display: none; }
.q > summary:hover { background: var(--bg); }
.q[open] > summary { background: var(--bg); border-bottom: 1px solid var(--line); }
.q .score { font-weight: 600; font-variant-numeric: tabular-nums; }
.qbody { padding: 4px 20px 20px; }

/* Verdict box: where the problem probably is */
.verdict { border-left: 4px solid; border-radius: 6px; padding: 10px 14px; margin-top: 16px; }
.verdict.pass { border-color: var(--good); background: var(--good-soft); }
.verdict.check { border-color: var(--check); background: var(--check-soft); }
.verdict.fail { border-color: var(--bad); background: var(--bad-soft); }

/* Expected vs actual answer */
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.compare > div { border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; background: var(--panel); }
.compare .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
.answer-text { white-space: pre-wrap; }

/* Token parts table inside a question */
.parts { max-width: 620px; }
.parts td { padding: 4px 10px; border: none; }
.parts td:nth-child(2), .parts td:nth-child(3) { text-align: right; font-variant-numeric: tabular-nums; }
.parts tr.sum td { border-top: 1px solid var(--line); font-weight: 600; }

/* Retrieved chunks: folded, click to read */
.chunk { border: 1px solid var(--line); border-radius: 8px; margin: 6px 0; background: var(--panel); }
.chunk > summary { cursor: pointer; padding: 6px 12px; font-size: 14px; }
.chunk pre { margin: 0; padding: 8px 12px 12px; white-space: pre-wrap; font-size: 12.5px; max-height: 280px; overflow: auto;
             border-top: 1px solid var(--line); }
.tag { font-size: 12px; font-weight: 600; color: var(--good); margin-left: 6px; }

/* Agent reports: tool names */
code.tool { font-size: 12.5px; padding: 1px 6px; border-radius: 4px; background: var(--bg); border: 1px solid var(--line); }

/* Links (to Phoenix traces) */
a { color: #4c8bf5; text-decoration: none; }
a:hover { text-decoration: underline; }

/* --- Phones: stack instead of side by side ------------------------------------ */
@media (max-width: 720px) {
  .qhead { display: none; }
  .q > summary { display: flex; flex-wrap: wrap; gap: 4px 14px; }
  .q > summary span:nth-child(2) { display: none; }
  .q > summary span:nth-child(3) { flex-basis: calc(100% - 80px); }
  .q > summary span:nth-child(n+4) { font-size: 13px; }
  .q > summary span[data-label]::before { content: attr(data-label) " "; color: var(--muted); font-weight: 400; }
  .compare { grid-template-columns: 1fr; }
  .scores th:nth-child(4), .scores td:nth-child(4) { display: none; }  /* "If low, look at": too narrow on a phone */
}
```

</details>

### 6.5 The RAG eval files

The RAG eval folder:

**`eval/rag/__init__.py`**
```python
"""
Doc 1: evaluate the RAG pipeline. The LangGraph agent eval (Doc 4) reuses this code.

    questions.jsonl  the test set: question, reference answer, expected paper
    evaluate.py      run the eval                          (python -m eval.rag.evaluate)
    tokens.py        split the answer call's tokens into instructions / context / question / output
    diagnose.py      pass / check / fail per question, and the likely cause
    report.py        write results/report.html
    results/         output of the last run: results.json + report.html
"""
```

`tokens.py` counts tokens with gpt-oss's tokenizer and splits the answer call's input into instructions, context, question and chat format (the table in 6.2):

**`eval/rag/tokens.py`**
```python
"""
Split the tokens of the RAG answer call into its parts.

Phoenix shows the exact input and output tokens of every call, but not what
the input is made of. So we count each part ourselves with gpt-oss's
tokenizer. What's left over is the chat format: the hidden system header and
role markers the server wraps around our prompt.
"""

import tiktoken

from app.rag.query import PROMPT, build_context

# gpt-oss uses the "o200k" tokenizer, the same one as OpenAI's recent models.
_tokenizer = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count how many tokens a text is for gpt-oss (roughly 1 token per 4 characters of English)."""
    return len(_tokenizer.encode(text))


def answer_tokens(question: str, out: dict) -> dict:
    """Split the tokens of the RAG answer call into its parts.

    Args:
        question: The question that was asked.
        out: The dict returned by `app.rag.query.answer()` (needs `contexts` and `usage`).

    Returns:
        A dict with:
            instructions  the fixed rules in PROMPT
            context       the retrieved chunks
            question      the user's question
            chat_format   tokens the server adds around the prompt (exact input - our 3 parts)
            input         exact input tokens reported by the server
            reasoning     output tokens the model spent "thinking" (not shown to the user)
            answer        output tokens of the visible answer
            output        exact output tokens reported by the server (reasoning + answer)
            total         input + output
    """
    usage = out["usage"]
    parts = {
        # PROMPT with empty placeholders = only the fixed instruction text.
        "instructions": count_tokens(PROMPT.format(context="", question="")),
        "context": count_tokens(build_context(out["contexts"])),
        "question": count_tokens(question),
    }
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)

    return {
        **parts,
        "chat_format": max(input_tokens - sum(parts.values()), 0),
        "input": input_tokens,
        "reasoning": reasoning,
        "answer": output_tokens - reasoning,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }
```

`diagnose.py` decides, for each question, pass / check / fail and the most likely cause. It turns the "low score means fix…" column of the table in 6.1 into rules:

**`eval/rag/diagnose.py`**
```python
"""
Decide, for one RAG question, whether it passed and where to look if it didn't.

Each score points at one part of the pipeline (see the metric table in Doc 1):
    retrieval_hit, context_recall  ->  retrieval (the chunks)
    faithfulness                   ->  the LLM (did it stick to the chunks?)
    answer_accuracy                ->  anything; the other three narrow it down

The rules combine them into one likely cause. It's a hint for where to look,
not a verdict: LLM judges make mistakes too.
"""

from eval.shared.status import BAD, GOOD, Diagnosis, level


def says_dont_know(answer: str) -> bool:
    """True if the answer refuses ("I don't know")."""
    text = answer.lower().replace("’", "'")  # the LLM sometimes writes a curly apostrophe
    return "don't know" in text or "do not know" in text


def diagnose_rag(row: dict) -> Diagnosis:
    """Decide the status of one RAG question and the most likely cause of a low score.

    Args:
        row: One result row from eval.rag.evaluate: the 4 scores, the answer and the expected source.

    Returns:
        A Diagnosis.
    """
    hit, recall = row["retrieval_hit"], row["context_recall"]
    faith, acc = row["faithfulness"], row["answer_accuracy"]

    # Special case: the papers don't contain the answer (source is None),
    # so "I don't know" is the right reply, whatever the judge scores say.
    if row["source"] is None:
        if says_dont_know(row["answer"]):
            return Diagnosis("pass", "Correct refusal",
                             "The papers don't contain this, and the system said so. "
                             "Faithfulness is often 0 here: a refusal has no claims for the judge to check.")
        return Diagnosis("fail", "LLM",
                         "Hallucination: the papers don't contain this, but the system answered anyway.")

    # The status comes from the lowest score.
    worst = level(min(hit, recall, faith, acc))
    status = {"good": "pass", "check": "check", "bad": "fail"}[worst]
    if status == "pass":
        return Diagnosis("pass")

    # The rules, from the most specific to the most general.
    if hit == 0:
        return Diagnosis(status, "Retrieval",
                         f"The expected paper ({row['source']}) is not among the retrieved chunks.")
    if recall < BAD and acc < GOOD:
        return Diagnosis(status, "Retrieval",
                         "The chunks miss facts the answer needs, so the LLM could not answer fully.")
    if recall < BAD:
        return Diagnosis(status, "Judge?",
                         "The answer is right, but the recall judge says the chunks lack the facts. "
                         "Check the chunks: the LLM may have used its own memory, or the judge is too strict.")
    if faith < BAD and acc >= GOOD:
        return Diagnosis(status, "Judge?",
                         "The answer is right and the chunks had the facts, but the faithfulness judge "
                         "found unsupported claims. Usually a judge error.")
    if acc < GOOD and faith < GOOD:
        return Diagnosis(status, "LLM",
                         "The answer doesn't match the reference and adds claims that are not in the chunks.")
    if acc < GOOD:
        return Diagnosis(status, "LLM or judge",
                         "The chunks had the facts and the answer stuck to them, but it doesn't match "
                         "the reference. Compare the two yourself.")
    return Diagnosis(status, "Minor", "Small deductions only. Read the answer to see if they matter.")
```

`report.py` builds the HTML page. Like the other page files, copy it as it is:

<details><summary><b><code>eval/rag/report.py</code></b> (click to show)</summary>

**`eval/rag/report.py`**
```python
"""
The HTML report for the RAG eval (Doc 1): eval/rag/results/report.html
and for the LangGraph agent eval (Doc 4): eval/langgraph_agent/results/report.html

Page layout, top to bottom:
    1. Summary   passed / check / failed, the mean of each score, token totals
    2. Questions "Needs attention" first, then "Passed". Click a row to see the
                 diagnosis, expected vs actual answer, the token split of its
                 context window, the retrieved chunks, and a link to its trace in Phoenix.

The judge's tokens, and every call's timing, are in Phoenix (project "rag-eval").
"""

from pathlib import Path

from app.shared import config
from eval.rag.diagnose import diagnose_rag
from eval.shared.status import GOOD, level
from eval.shared.html_page import (esc, legend, n, page, phoenix_panel, question_list, question_row,
                                   stat, token_bar, trace_link, verdict)

# The 4 scores: key -> (column name, what it checks, a low score points to)
METRICS = {
    "retrieval_hit": ("Hit", "Is the expected paper among the 5 retrieved chunks?", "retrieval"),
    "context_recall": ("Recall", "Do the chunks contain the facts the reference answer needs?", "retrieval (chunking, k)"),
    "faithfulness": ("Faith", "Is every claim in the answer backed by the chunks?", "the LLM (hallucination)"),
    "answer_accuracy": ("Acc", "Does the answer match the reference answer?", "anything: check the other three"),
}

# The parts of one answer call, in the order the model sees them: key -> (name, colour)
PARTS = {
    "instructions": ("Instructions", "#8e9aaf"),
    "context": ("Context (chunks)", "#4c8bf5"),
    "question": ("Question", "#a371f7"),
    "chat_format": ("Chat format", "#c9ced6"),
    "reasoning": ("Reasoning (output)", "#e3a33b"),
    "answer": ("Answer (output)", "#3fb68b"),
}

# What a part is, shown next to it in a question's token table.
PART_HELP = {
    "chat_format": "added by the server around every prompt (system header, role markers)",
    "reasoning": "gpt-oss thinks before it answers: generated and billed, but not shown",
}


def parts_of(tokens: dict) -> list[tuple[str, str, float]]:
    """(name, colour, tokens) for each part of an answer call, for token_bar() and legend()."""
    return [(name, color, tokens[key]) for key, (name, color) in PARTS.items()]


def summary(results: list[dict], statuses: list[str], project: str) -> str:
    """Pass/check/fail counts, a table of mean scores, and the answer calls' tokens."""
    count = len(results)

    def mean(key):
        return sum(r[key] for r in results) / count

    def total(key):
        return sum(r["tokens"][key] for r in results)

    score_rows = "".join(
        f'<tr><td><b>{name}</b></td><td class="score {level(mean(key))}">{mean(key):.2f}</td>'
        f'<td>{checks}</td><td class="muted">{points_to}</td></tr>'
        for key, (name, checks, points_to) in METRICS.items())

    answer_total = total("total")
    average_call = {key: total(key) / count for key in PARTS}

    return f"""
<div class="stats">
  {stat(statuses.count("pass"), f"passed (all scores ≥ {GOOD})", "good")}
  {stat(statuses.count("check"), "to check (a score 0.5–0.8)", "check")}
  {stat(statuses.count("fail"), "failed (a score below 0.5)", "bad")}
</div>

<h2>Scores</h2>
<div class="panel" style="padding:4px 10px">
<table class="scores">
  <tr><th>Score</th><th>Mean</th><th>What it checks</th><th>If low, look at</th></tr>
  {score_rows}
</table>
</div>

<h2>Tokens</h2>
<div class="stats" style="margin-top:0">
  {stat(n(answer_total), "tokens to answer all questions")}
  {stat(n(answer_total / count), "tokens per answer call")}
</div>
<div class="panel" style="margin-top:12px">
  <p><b>The average answer call: {n(answer_total / count)} tokens</b>
     <span class="muted">= {answer_total / count / config.CONTEXT_WINDOW:.1%} of the {n(config.CONTEXT_WINDOW)}-token context window</span></p>
  {token_bar(parts_of(average_call))}
  {legend(parts_of(average_call))}
</div>
{phoenix_panel(project)}"""


def token_table(tokens: dict) -> str:
    """The token split of one answer call: input parts, then output parts, then the total."""
    def row(key):
        name, color = PARTS[key]
        note = f'<span class="muted small"> · {PART_HELP[key]}</span>' if key in PART_HELP else ""
        return (f'<tr><td><i class="dot" style="background:{color}"></i>{name}{note}</td>'
                f'<td>{n(tokens[key])}</td><td class="muted">{tokens[key] / tokens["total"]:.0%}</td></tr>')

    return f"""
{token_bar(parts_of(tokens))}
<table class="parts">
  {"".join(row(key) for key in ["instructions", "context", "question", "chat_format"])}
  <tr class="sum"><td>Input</td><td>{n(tokens["input"])}</td><td></td></tr>
  {"".join(row(key) for key in ["reasoning", "answer"])}
  <tr class="sum"><td>Output</td><td>{n(tokens["output"])}</td><td></td></tr>
  <tr class="sum"><td>Answer call total <span class="muted small">· {tokens["total"] / config.CONTEXT_WINDOW:.1%} of the context window</span></td>
      <td>{n(tokens["total"])}</td><td></td></tr>
</table>
<p class="muted small">Phoenix shows the same call, plus the judge's calls that scored it.</p>"""


def chunk_list(row: dict) -> str:
    """The retrieved chunks, folded. Chunks from the expected paper are marked."""
    items = ""
    for i, chunk in enumerate(row["contexts"], start=1):
        mark = '<span class="tag">✓ expected paper</span>' if chunk["source"] == row["source"] else ""
        items += (f'<details class="chunk"><summary>#{i} · {esc(chunk["source"])} p.{chunk["page"]} '
                  f'<span class="muted">· similarity {chunk["score"]:.2f}</span>{mark}</summary>'
                  f'<pre>{esc(chunk["text"])}</pre></details>')
    return items


def rag_row(num: int, row: dict) -> tuple[str, str]:
    """One question row of the RAG report. Returns (status, html)."""
    diag = diagnose_rag(row)

    cells = ""
    for key, (name, _, _) in METRICS.items():
        text = ("yes" if row[key] else "no") if key == "retrieval_hit" else f"{row[key]:.2f}"
        cells += f'<span class="score {level(row[key])}" data-label="{name}">{text}</span>'
    cells += f'<span class="num muted" data-label="Tokens">{n(row["tokens"]["total"])}</span>'

    expected_paper = esc(row["source"]) if row["source"] else "none (the papers don't contain the answer)"

    # The LangGraph agent (Doc 4) may search twice, with a rewritten query.
    searches = row.get("searches", 1)
    search_note = (f'<p class="muted small">Searched {searches} times; the last query was rewritten to: '
                   f'<code>{esc(row["query"])}</code></p>' if searches > 1 else "")

    details = f"""
    {verdict(diag)}
    <h3>Answer</h3>
    <div class="compare">
      <div><div class="label">Expected</div><div class="answer-text">{esc(row["reference"])}</div></div>
      <div><div class="label">Got</div><div class="answer-text">{esc(row["answer"])}</div></div>
    </div>
    <h3>Context window of the answer call</h3>
    {token_table(row["tokens"])}
    <h3>Retrieved chunks</h3>
    <p class="muted small">Expected paper: {expected_paper}. Click a chunk to read it.</p>
    {search_note}
    {chunk_list(row)}
    <h3>Trace</h3>
    <p>The answer call and the judge's calls, with tokens and timing:
       {trace_link(row, "open this question's trace in Phoenix ↗")}</p>"""
    return diag.status, question_row(num, row["question"], diag, cells, details)


def write_rag_report(results: list[dict], path: Path, workers: int | str = "?",
                     title: str = "RAG", project: str = "rag-eval") -> None:
    """Write the HTML report for a RAG eval run (plain RAG, or the LangGraph agent).

    Args:
        results: The rows from `eval.rag.evaluate`, each with its 4 scores, a `tokens` dict and `trace_id`.
        path: Where to write the .html file.
        workers: How many questions ran in parallel (shown under the title).
        title: Which system was tested, e.g. "RAG" or "LangGraph agent".
        project: Its Phoenix project, e.g. "rag-eval".
    """
    rows = [rag_row(num, row) for num, row in enumerate(results, start=1)]
    statuses = [status for status, _ in rows]
    columns = [name for name, _, _ in METRICS.values()] + ["Tokens"]

    body = (summary(results, statuses, project)
            + question_list("Needs attention", "Open by default. The coloured box says where the problem "
                            "probably is: retrieval, the LLM, or the judge. It's a hint, not a verdict: "
                            "read the answer and the chunks to decide.",
                            columns, [html for status, html in rows if status != "pass"])
            + question_list("Passed", "Click a question to see its details.",
                            columns, [html for status, html in rows if status == "pass"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(f"{title} eval report", f"{len(results)} questions · {workers} in parallel", body),
                    encoding="utf-8")
```

</details>

`evaluate.py` is the eval itself: for each question it runs `answer()`, checks retrieval, asks the judge for the three scores and splits the answer's tokens. Each question is wrapped in one trace, so in Phoenix you see its answer and its judge calls together. Then it prints the averages and saves `results.json` and `report.html`:

**`eval/rag/evaluate.py`**
```python
"""
Evaluate the RAG pipeline (Doc 1) on the questions in `eval/rag/questions.jsonl`.

For each question we run `answer()` and then measure 4 things:

    retrieval_hit    Did search find a chunk from the right PDF?        (plain code, no LLM)
    context_recall   Do the found chunks contain the facts we need?     (LLM judge)
    faithfulness     Is the answer backed by the chunks, not invented?  (LLM judge)
    answer_accuracy  Does the answer match the reference answer?        (LLM judge)

Why these four? When a question fails, they show where it went wrong:
    - low hit / recall      -> retrieval is the problem (wrong chunks found)
    - low faithfulness      -> the LLM made things up
    - low accuracy only     -> the LLM misread good chunks, or the judge is wrong

An "LLM judge" is an LLM that grades another LLM's work. RAGAS provides the
judge prompts. Judges are noisy, so read the failing rows yourself.

Run: uv run python -m eval.rag.evaluate
Output: a START and a DONE line per question (with timestamps, so you can see
WORKERS questions running in parallel), the token use of each question, the
averages, and eval/rag/results/ with results.json and report.html (open it in a browser).
Every question is also one trace in Phoenix (project "rag-eval"): the answer call
and all judge calls, with their tokens. The report links to each one.

The LangGraph agent eval (Doc 4) reuses main() below with its own system and names.
"""

import asyncio
import sys
import time
from pathlib import Path

from ragas.metrics.collections import AnswerAccuracy, ContextRecall, Faithfulness

from app.rag.query import answer
from app.shared.tracing import current_trace_id, start_tracing
from eval.rag.report import write_rag_report
from eval.rag.tokens import answer_tokens
from eval.shared.judge import judge_llm
from eval.shared.questions import load_questions, save_results

# How many questions are evaluated at the same time. LM Studio handles 4
# requests in parallel, so more workers would only wait in its queue.
WORKERS = 4

HERE = Path(__file__).parent                        # eval/rag/
QUESTIONS_FILE = HERE / "questions.jsonl"
RESULTS_FILE = HERE / "results" / "results.json"
REPORT_FILE = HERE / "results" / "report.html"


async def judge(row: dict, metrics: dict) -> dict:
    """Score one answered question with the three RAGAS judge metrics.

    Args:
        row: A question dict that already has `answer` and `contexts`.
        metrics: The RAGAS metric objects created in `main()`.

    Returns:
        `{"context_recall", "faithfulness", "answer_accuracy"}`, each from 0 to 1.
    """
    contexts = [chunk["text"] for chunk in row["contexts"]]  # RAGAS wants only the text

    # The three judge calls don't depend on each other, so run them together.
    recall, faith, acc = await asyncio.gather(
        # Compare the reference answer with the chunks: was everything needed found?
        metrics["recall"].ascore(
            user_input=row["question"], retrieved_contexts=contexts, reference=row["reference"]),
        # Compare the answer with the chunks: is each claim backed by them?
        metrics["faith"].ascore(
            user_input=row["question"], response=row["answer"], retrieved_contexts=contexts),
        # Compare the answer with the reference: is it correct?
        metrics["acc"].ascore(
            user_input=row["question"], response=row["answer"], reference=row["reference"]),
    )
    return {
        "context_recall": recall.value,
        "faithfulness": faith.value,
        "answer_accuracy": acc.value,
    }


def log(t0: float, message: str) -> None:
    """Print a message with the seconds since the eval started, e.g. `[ 12.3s] ...`.

    The timestamps show the parallel work: several START lines at 0.0s, and a
    new START right after each DONE (a slot freed up, the next question takes it).
    """
    print(f"[{time.perf_counter() - t0:5.1f}s] {message}")


# How many questions are running right now. Only shown in the log, to make the
# parallel work visible. A plain global is safe: asyncio runs on one thread and
# switches tasks only at an `await`, never in the middle of `running += 1`.
running = 0


async def evaluate_one(num: int, row: dict, metrics: dict, limit: asyncio.Semaphore, t0: float,
                       tracer, answer_fn) -> dict:
    """Answer one question, check retrieval, run the judges, and log START and DONE.

    Args:
        num: Question number (1, 2, ...), only used in the log.
        row: One question dict from the question file.
        metrics: The RAGAS metric objects.
        limit: Semaphore that lets at most WORKERS questions run at once.
        t0: Time the eval started, for the log timestamps.
        tracer: From start_tracing(); groups this question's calls into one Phoenix trace.
        answer_fn: The system to test: plain RAG's answer(), or the LangGraph agent's.

    Returns:
        The same row with the answer, the contexts, the latency and all scores added.
    """
    global running
    async with limit:  # wait here until one of the WORKERS slots is free
        # One trace per question: the answer call and the judge calls all land under it.
        with tracer.start_as_current_span(f"Q{num}: {row['question']}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")   # how Phoenix labels this step
            span.set_attribute("input.value", row["question"])
            row["trace_id"] = current_trace_id()  # so the report can link to it
            running += 1
            log(t0, f"START Q{num:<2} (running: {running})  {row['question'][:55]}")

            # 1. Run the RAG pipeline and time it. answer() is normal (sync) code,
            #    so run it in a thread to avoid blocking the other questions.
            start = time.perf_counter()
            out = await asyncio.to_thread(answer_fn, row["question"])
            row["answer"] = out["answer"]
            row["contexts"] = out["contexts"]
            row["searches"] = out.get("searches", 1)          # the LangGraph agent may search twice
            row["query"] = out.get("query", row["question"])  # ...with a rewritten query
            row["latency_s"] = round(time.perf_counter() - start, 1)
            row["tokens"] = answer_tokens(row["question"], out)  # instructions / context / question / output
            span.set_attribute("output.value", row["answer"])

            # 2. Retrieval check without an LLM: is the expected PDF among the found
            #    chunks? Questions with source=None (no single right PDF) always count as a hit.
            found_sources = {chunk["source"] for chunk in out["contexts"]}
            row["retrieval_hit"] = float(row["source"] is None or row["source"] in found_sources)

            # 3. LLM-judge scores.
            row |= await judge(row, metrics)

            tokens = row["tokens"]
            running -= 1
            log(t0, f"DONE  Q{num:<2} (running: {running})  hit={row['retrieval_hit']:.0f} "
                    f"recall={row['context_recall']:.2f} faith={row['faithfulness']:.2f} "
                    f"acc={row['answer_accuracy']:.2f}  answer took {row['latency_s']}s")
            log(t0, f"      Q{num:<2} tokens: answer {tokens['input']:,} in (context {tokens['context']:,}) "
                    f"+ {tokens['output']:,} out (reasoning {tokens['reasoning']:,})")
            return row


async def main(answer_fn=answer, results_dir=RESULTS_FILE.parent, title="RAG", project="rag-eval"):
    """Evaluate all questions, print the averages, and save the results and the report.

    The arguments let Doc 4 reuse this same eval for the LangGraph agent.

    Args:
        answer_fn: The system to test. Takes a question, returns `{"answer", "contexts", "usage"}`.
        results_dir: Folder for results.json and report.html.
        title: Name of the system, shown in the HTML report.
        project: The Phoenix project the traces go to.
    """
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    tracer = start_tracing(project, judge=True)  # record the system and the judge in Phoenix

    # One judge shared by the three metrics.
    llm = judge_llm()
    metrics = {
        "recall": ContextRecall(llm=llm),
        "faith": Faithfulness(llm=llm),
        "acc": AnswerAccuracy(llm=llm),
    }
    rows = load_questions(QUESTIONS_FILE)

    # Start every question at once. The semaphore keeps only WORKERS running,
    # and gather() returns the results in the original question order.
    print(f"{len(rows)} questions, {WORKERS} at a time\n")
    t0 = time.perf_counter()
    limit = asyncio.Semaphore(WORKERS)
    results = await asyncio.gather(
        *(evaluate_one(num, row, metrics, limit, t0, tracer, answer_fn) for num, row in enumerate(rows, start=1)))
    print(f"\ntotal time: {time.perf_counter() - t0:.0f}s with {WORKERS} workers")

    # The average of each score is the number that goes in the README results table.
    keys = ["retrieval_hit", "context_recall", "faithfulness", "answer_accuracy", "latency_s", "searches"]
    means = [f"{key}={sum(r[key] for r in results) / len(results):.2f}" for key in keys]
    print("\nMEAN  " + "  ".join(means))

    # Answer-call tokens over all questions. The judge's tokens, and the total of every
    # call, are in Phoenix: project "rag-eval" shows them at the top of its page.
    def total(key):
        return sum(r["tokens"][key] for r in results)
    print(f"TOKENS  answer calls: {total('input'):,} in + {total('output'):,} out"
          f"  |  judge + everything: see Phoenix, project {project}")

    # Save every row (answer, chunks, scores, tokens) so you can read the failing ones later,
    # and the same data as an HTML page for the browser.
    results_dir = Path(results_dir)
    save_results(results, results_dir / "results.json")
    write_rag_report(results, results_dir / "report.html", WORKERS, title, project)
    print(f"\nreport: {(results_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
```

`rebuild_reports.py` rebuilds the reports from the saved results, without calling the LLM. Handy when you change the report code:

**`eval/rebuild_reports.py`**
```python
"""
Rebuild the HTML reports from the last saved results, without re-running any eval.
No LLM calls, so it takes a second. Handy when you change the report code.

Run: uv run python -m eval.rebuild_reports
"""

import json

from eval.rag.evaluate import REPORT_FILE, WORKERS
from eval.rag.report import write_rag_report

# The RAG-style evals (Doc 1 + 4): results folder -> title shown in the report, Phoenix project.
for folder, title, project in [("rag", "RAG", "rag-eval"),
                               ("langgraph_agent", "LangGraph agent", "langgraph-eval")]:
    results_file = REPORT_FILE.parents[2] / folder / "results" / "results.json"   # eval/<folder>/results/
    if results_file.exists():
        report_file = results_file.with_name("report.html")
        write_rag_report(json.loads(results_file.read_text(encoding="utf-8")), report_file, WORKERS, title, project)
        print(f"report: {report_file.resolve()}")

# The agent evals (Doc 2 + 3): results folder -> title shown in the report, Phoenix project.
for folder, title, project in [("single_agent", "Single agent", "single-agent-eval"),
                               ("multi_agent", "Multi-agent", "multi-agent-eval")]:
    results_file = REPORT_FILE.parents[2] / folder / "results" / "results.json"   # eval/<folder>/results/
    if not results_file.exists():
        continue  # that eval hasn't been run yet
    # Imported here, not at the top: the agent report is only written in Doc 2.
    from eval.single_agent.report import write_agent_report

    report_file = results_file.with_name("report.html")
    write_agent_report(json.loads(results_file.read_text(encoding="utf-8")), report_file, title, project)
    print(f"report: {report_file.resolve()}")
```

### 6.6 Run it

```bash
uv run python -m eval.rag.evaluate     # ~1.5 min; you may see library warnings: ignore them
start eval/rag/results/report.html     # open the report in your browser (Git Bash on Windows)
```

**Parallel questions:** `WORKERS = 4` runs 4 *different* questions at once, about 2× faster overall. The log shows it:

```
[  0.0s] START Q1  (running: 1)  How many attention heads does the base Transformer mode
[  0.0s] START Q2  (running: 2)  What BLEU score did the big Transformer get on WMT 2014
[  0.0s] START Q3  (running: 3)  Why does the Transformer need positional encodings?
[  0.0s] START Q4  (running: 4)  Which retriever does the RAG model use?
[ 22.0s] DONE  Q1  (running: 3)  hit=1 recall=1.00 faith=1.00 acc=1.00  answer took 4.2s
[ 22.9s] START Q5  (running: 4)  What is the difference between RAG-Sequence and RAG-Tok
```

Four questions start at 0.0s. Each time one finishes, a slot frees up and the next question takes it, so "running" stays at 4 until the queue is empty. Most of a question's time goes to the judge calls; "answer took" is only the RAG part. Each question's `latency_s` goes *up*, because 4 questions share one GPU. To measure single-user speed, set `WORKERS = 1`: the log then shows START, DONE, START, DONE, one at a time.

When this was tested, the means were about `retrieval_hit=1.00 context_recall=0.70–0.91 faithfulness=0.82–0.91 answer_accuracy=0.86–0.91`, about 90,000 tokens per run (85% of it the judge: see the `rag-eval` project in Phoenix). They move between runs because the judge is noisy. Write your numbers in the table in [README.md](../README.md).

### 6.7 Read the report

Open `eval/rag/results/report.html`:
- **Top:** how many questions passed, need a check, or failed; the mean of each score with what it checks; the answer tokens and the split of an average answer call.
- **Needs attention:** every problem question, already open, with a coloured box saying where the problem probably is (retrieval, the LLM, or the judge), the expected answer next to the actual answer, the token split of its context window, the retrieved chunks (✓ marks chunks from the expected paper), and a link to its trace in Phoenix.
- **Passed:** the rest, folded. Click one to open it.

The coloured box is a hint for where to look, not a verdict. **Read the failures, not just the mean.** In the test run:
- **"Which generator model does RAG use?"** The answer said "BART", but the reference says "BART-large", and none of the 5 retrieved chunks contained "large". That's a **retrieval** miss (`context_recall=0`), not an LLM mistake.
- **"How many attention heads…"** The answer "eight (8) heads [p.5]" was correct, yet `faithfulness=0`. That's a **judge error**. LLM judges are noisy, so a single low score is a reason to look, not a verdict.

---

## 7. Experiment: change one thing (30 min)

This is how improvement works in real teams: **change one setting, re-run the eval, keep the change only if the numbers improve.** Try one or two:

| Change | Where | Expectation |
|---|---|---|
| `k=5` → `k=8` | `answer(..., k=8)` in `app/rag/query.py` | higher context_recall, slower, more noise for the LLM |
| `chunk_size=1000` → `500` | `app/rag/ingest.py`, then re-run ingest | more precise chunks, less context in each |
| Remove the "I don't know" line from the prompt | `app/rag/query.py` | watch the GPT-4 question start hallucinating |

Record each run as a row in the README results table.

---

## Checklist

- [x] I can explain ingest vs query, and what a chunk, an embedding and a vector DB are.
- [x] I saw my chunks in the Qdrant dashboard, and a RAG trace (search + LLM call) in Phoenix.
- [x] I can say which metric points to retrieval and which to generation.
- [x] I opened the HTML report and can say what each part shows, including the token split of a context window.
- [x] I read at least two failing questions and decided whether retrieval, the LLM or the judge was wrong.
- [x] I ran one experiment and recorded the before/after numbers.

**Next:** [Doc 2: AI agent and its evaluation](02-ai-agent-and-evaluation.md)
