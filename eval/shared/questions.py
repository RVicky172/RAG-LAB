"""
Read a question file and save results: the same for every eval.
"""

import json
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    """Read a question file. JSONL means one JSON object per line.

    Args:
        path: Path to the .jsonl file.

    Returns:
        One dict per question, e.g. `{"question", "reference", ...}`.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]  # skip empty lines


def save_results(rows: list[dict], path: Path) -> None:
    """Save every result row as readable JSON, creating the folder if needed.

    Args:
        rows: The result rows (question, answer, scores, tokens, ...).
        path: Where to write the .json file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
