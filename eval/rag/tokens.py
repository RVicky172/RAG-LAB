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
