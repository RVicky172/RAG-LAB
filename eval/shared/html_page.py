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
