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
