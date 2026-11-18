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
