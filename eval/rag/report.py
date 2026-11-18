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
