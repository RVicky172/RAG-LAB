"""
The state and the steps (nodes) of the self-correcting RAG graph (Doc 4).

In LangGraph:
    - the STATE is one dict that every step can read. It flows through the graph.
    - a NODE is a plain function: it gets the state and returns only the keys it
      changes. LangGraph merges them into the state.
    - an EDGE says which node runs next. A CONDITIONAL edge is a function that
      looks at the state and returns the name of the next node.

The idea: plain RAG (Doc 1) searches once and answers with whatever it found.
This graph first asks the LLM "do these chunks answer the question?". If not,
it rewrites the search query and searches again, once.
"""

from typing import TypedDict

from app.rag.query import PROMPT, build_context, search
from app.shared.llm import get_llm

# At most this many searches per question: the first one plus one retry.
# Without a limit, a question the papers can't answer would loop forever.
MAX_SEARCHES = 2

# How many chunks the answer gets, as in plain RAG.
K = 5

# One LLM connection shared by all the steps.
llm = get_llm()


class State(TypedDict, total=False):
    """Everything the steps share. total=False: a key only exists once a step has set it."""

    question: str        # the user's question (never changes)
    query: str           # what we search with: the question at first, maybe rewritten later
    chunks: list[dict]   # the best chunks found so far
    searches: int        # how many searches so far
    relevant: bool       # grade's verdict: do the chunks answer the question?
    answer: str          # the final answer
    usage: dict          # token usage of the answer call (the eval report uses it)


# --- The steps ---------------------------------------------------------------------

def best_chunks(chunks: list[dict]) -> list[dict]:
    """The K most similar chunks, without duplicates (two searches can find the same chunk)."""
    unique = {(c["source"], c["page"], c["text"]): c for c in chunks}.values()
    return sorted(unique, key=lambda c: c["score"], reverse=True)[:K]


def retrieve(state: State) -> dict:
    """Search Qdrant with the current query (the question itself on the first try).

    On a retry, keep the best chunks of BOTH searches: if the new query finds
    worse chunks, the good ones from the first search aren't lost.
    """
    query = state.get("query") or state["question"]
    found = search(query, K)
    return {
        "query": query,
        "chunks": best_chunks(state.get("chunks", []) + found),
        "searches": state.get("searches", 0) + 1,
    }


GRADE_PROMPT = """Is the text below relevant to the question: does it contain
information that helps to answer it? Reply with one word: yes or no.

Question: {question}

Text:
{context}"""


def grade(state: State) -> dict:
    """Ask the LLM whether the chunks contain what the question needs."""
    prompt = GRADE_PROMPT.format(question=state["question"], context=build_context(state["chunks"]))
    reply = llm.invoke(prompt).content
    return {"relevant": reply.strip().lower().startswith("yes")}


REWRITE_PROMPT = """A search of AI research papers found nothing useful for this question.
Write one better search query: use other words and the likely technical terms.
The search compares meanings, not keywords, so write a short plain sentence:
no quotes, no AND / OR. Reply with the query only.

Question: {question}
Last query: {query}"""


def rewrite(state: State) -> dict:
    """Ask the LLM for a better search query."""
    prompt = REWRITE_PROMPT.format(question=state["question"], query=state["query"])
    query = llm.invoke(prompt).content.strip().strip('"')  # the LLM sometimes wraps it in quotes
    return {"query": query}


def generate(state: State) -> dict:
    """Answer from the chunks, with the same prompt as plain RAG (Doc 1)."""
    prompt = PROMPT.format(context=build_context(state["chunks"]), question=state["question"])
    reply = llm.invoke(prompt)
    return {"answer": reply.content, "usage": reply.usage_metadata or {}}


# --- The conditional edge ------------------------------------------------------------

def after_grade(state: State) -> str:
    """Decide what runs after grade: "generate" or "rewrite".

    Answer when the chunks are relevant, or when we're out of searches (then
    the prompt's "I don't know" rule takes over). Otherwise try a new query.
    """
    if state["relevant"] or state["searches"] >= MAX_SEARCHES:
        return "generate"
    return "rewrite"
