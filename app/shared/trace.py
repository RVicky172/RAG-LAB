"""
Print an agent's conversation (its "trace") one line per message.

Both agents use it when you run them from the terminal, so you can see the
loop: question -> tool call -> tool result -> answer. For the full picture
(tokens, timing, what each specialist did inside) open the trace in Phoenix.
"""


def print_trace(messages) -> None:
    """Print one line per message: its tool calls if it made any, else the start of its text.

    Args:
        messages: The `messages` list an agent's `run()` returns.
    """
    for m in messages:
        # Only AI messages have tool_calls, so getattr gives [] for the others.
        calls = [f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", [])]
        print(f"[{m.type}] {calls or m.content[:150]}")
