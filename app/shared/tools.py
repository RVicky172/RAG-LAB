"""
The tools our agents can call. Both the single agent and the multi-agent system use them.

@tool turns a normal Python function into something the LLM can call.
The LLM sees only the function NAME, its ARGUMENTS and its DOCSTRING, so each
docstring below is written as an instruction for the LLM, not for us.
Changing a docstring changes how the agent behaves.
"""

from langchain.tools import tool

from app.rag.query import build_context, search


@tool
def search_papers(query: str) -> str:
    """Search the research-paper knowledge base (Transformer, RAG, Lost in the Middle, RAGAS).
    Returns the 5 most relevant text chunks with their source file and page."""
    # Reuse the RAG search from Doc 1. A tool must return text, so format the
    # chunks the same way the RAG prompt does: "[file p.N]" + the chunk text.
    return build_context(search(query))


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '28.4 - 26.3' or '512 / 8'."""
    # Safety: the LLM chooses this input, so allow only digits and + - * / ( ) .
    # With everything else rejected, eval() can only do arithmetic.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    return str(eval(expression))
