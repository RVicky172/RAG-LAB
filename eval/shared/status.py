"""
Grading shared by every eval: score thresholds, colours and the pass/check/fail result.

Each eval has its own rules for WHY a question failed (rag/diagnose.py,
single_agent/diagnose.py); this file holds what they have in common.
"""

from dataclasses import dataclass

# Score thresholds, used for every colour and status in the report.
GOOD = 0.8  # score >= GOOD          -> good
BAD = 0.5   # BAD <= score < GOOD    -> check;  score < BAD -> bad


@dataclass
class Diagnosis:
    """The result for one question.

    status: "pass", "check" or "fail".
    area:   where the problem probably is, e.g. "Retrieval" or "LLM". Empty when it passed.
    reason: one plain sentence explaining it. Empty when there is nothing to say.
    """

    status: str
    area: str = ""
    reason: str = ""


def level(score: float) -> str:
    """Grade one score: "good", "check" or "bad"."""
    if score >= GOOD:
        return "good"
    return "check" if score >= BAD else "bad"
