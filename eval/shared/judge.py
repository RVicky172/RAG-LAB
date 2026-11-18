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
