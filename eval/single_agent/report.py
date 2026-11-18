"""
The HTML report for the agent evals (Doc 2 + 3):
eval/single_agent/results/report.html and eval/multi_agent/results/report.html

Page layout, top to bottom:
    1. Summary   passed / check / failed, tool match and goal accuracy, steps and time,
                 and where to find the tokens (Phoenix)
    2. Questions "Needs attention" first, then "Passed". Click a row to see the
                 diagnosis, expected vs actual answer, and expected vs called tools.
                 The full step-by-step trace is one click away, in Phoenix.
"""

from pathlib import Path

from eval.shared.html_page import (esc, page, phoenix_panel, question_list, question_row, stat,
                                   trace_link, verdict)
from eval.shared.status import level
from eval.single_agent.diagnose import diagnose_agent

# The 2 scores: key -> (name, what it checks, a low score points to)
METRICS = {
    "tool_match": ("Tool match", "Did it call exactly the expected tools (or specialists)?",
                   "behaviour: the system prompt or the tool docstrings"),
    "goal_accuracy": ("Goal", "Did the conversation reach the reference goal? (LLM judge reads the trace)",
                      "the answer, or the judge"),
}


def pills(names: list[str]) -> str:
    """Tool names as small labels, or 'none'."""
    if not names:
        return '<span class="muted">none</span>'
    return " ".join(f'<code class="tool">{esc(name)}</code>' for name in names)


def summary(results: list[dict], statuses: list[str], project: str) -> str:
    """Pass/check/fail counts, the two scores, steps and time, and a pointer to Phoenix."""
    count = len(results)

    def mean(key):
        return sum(r[key] for r in results) / count

    score_rows = "".join(
        f'<tr><td><b>{name}</b></td><td class="score {level(mean(key))}">{mean(key):.2f}</td>'
        f'<td>{checks}</td><td class="muted">{points_to}</td></tr>'
        for key, (name, checks, points_to) in METRICS.items())

    return f"""
<div class="stats">
  {stat(statuses.count("pass"), "passed (right tools, goal reached)", "good")}
  {stat(statuses.count("check"), "to check (goal reached the wrong way)", "check")}
  {stat(statuses.count("fail"), "failed (goal missed)", "bad")}
</div>

<h2>Scores</h2>
<div class="panel" style="padding:4px 10px">
<table class="scores">
  <tr><th>Score</th><th>Mean</th><th>What it checks</th><th>If low, look at</th></tr>
  {score_rows}
</table>
</div>
<div class="stats">
  {stat(f"{mean('steps'):.1f}", "tool calls per question")}
  {stat(f"{mean('latency_s'):.1f}s", "per question (one at a time)")}
</div>

<h2>Tokens and traces</h2>
{phoenix_panel(project)}"""


def agent_row(num: int, row: dict) -> tuple[str, str]:
    """One question row of the agent report. Returns (status, html)."""
    diag = diagnose_agent(row)

    cells = (f'<span class="score {level(row["tool_match"])}" data-label="Tools">{"yes" if row["tool_match"] else "no"}</span>'
             f'<span class="score {level(row["goal_accuracy"])}" data-label="Goal">{row["goal_accuracy"]:.2f}</span>'
             f'<span class="num" data-label="Steps">{row["steps"]}</span>'
             f'<span class="num muted" data-label="Time">{row["latency_s"]}s</span>'
             f'<span data-label="Trace">{trace_link(row, "open ↗")}</span>')

    details = f"""
    {verdict(diag)}
    <h3>Answer</h3>
    <div class="compare">
      <div><div class="label">Expected</div><div class="answer-text">{esc(row["reference"])}</div></div>
      <div><div class="label">Got</div><div class="answer-text">{esc(row["answer"])}</div></div>
    </div>
    <h3>Tools</h3>
    <div class="compare">
      <div><div class="label">Expected</div>{pills(row["expected_tools"])}</div>
      <div><div class="label">Called, in order</div>{pills(row["tools_called"])}</div>
    </div>
    <h3>Trace</h3>
    <p>Every step (LLM calls, tool calls with their inputs and results, the judge) with tokens and timing:
       {trace_link(row, "open this question's trace in Phoenix ↗")}</p>"""
    return diag.status, question_row(num, row["question"], diag, cells, details)


def write_agent_report(results: list[dict], path: Path, title: str, project: str) -> None:
    """Write the HTML report for an agent eval run.

    Args:
        results: The rows from `eval.single_agent.evaluate`, each with its scores and `trace_id`.
        path: Where to write the .html file.
        title: Which system was tested, e.g. "Single agent" or "Multi-agent".
        project: Its Phoenix project, e.g. "single-agent-eval".
    """
    rows = [agent_row(num, row) for num, row in enumerate(results, start=1)]
    statuses = [status for status, _ in rows]
    columns = ["Tools", "Goal", "Steps", "Time", "Trace"]

    body = (summary(results, statuses, project)
            + question_list("Needs attention", "Open by default. The coloured box says where the problem "
                            "probably is: the agent's behaviour, its answer, or the judge. It's a hint, not "
                            "a verdict: open the trace to decide.",
                            columns, [html for status, html in rows if status != "pass"])
            + question_list("Passed", "Click a question to see its details.",
                            columns, [html for status, html in rows if status == "pass"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(f"{title} eval report", f"{len(results)} questions · one at a time", body),
                    encoding="utf-8")
