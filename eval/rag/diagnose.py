"""
Decide, for one RAG question, whether it passed and where to look if it didn't.

Each score points at one part of the pipeline (see the metric table in Doc 1):
    retrieval_hit, context_recall  ->  retrieval (the chunks)
    faithfulness                   ->  the LLM (did it stick to the chunks?)
    answer_accuracy                ->  anything; the other three narrow it down

The rules combine them into one likely cause. It's a hint for where to look,
not a verdict: LLM judges make mistakes too.
"""

from eval.shared.status import BAD, GOOD, Diagnosis, level


def says_dont_know(answer: str) -> bool:
    """True if the answer refuses ("I don't know")."""
    text = answer.lower().replace("’", "'")  # the LLM sometimes writes a curly apostrophe
    return "don't know" in text or "do not know" in text


def diagnose_rag(row: dict) -> Diagnosis:
    """Decide the status of one RAG question and the most likely cause of a low score.

    Args:
        row: One result row from eval.rag.evaluate: the 4 scores, the answer and the expected source.

    Returns:
        A Diagnosis.
    """
    hit, recall = row["retrieval_hit"], row["context_recall"]
    faith, acc = row["faithfulness"], row["answer_accuracy"]

    # Special case: the papers don't contain the answer (source is None),
    # so "I don't know" is the right reply, whatever the judge scores say.
    if row["source"] is None:
        if says_dont_know(row["answer"]):
            return Diagnosis("pass", "Correct refusal",
                             "The papers don't contain this, and the system said so. "
                             "Faithfulness is often 0 here: a refusal has no claims for the judge to check.")
        return Diagnosis("fail", "LLM",
                         "Hallucination: the papers don't contain this, but the system answered anyway.")

    # The status comes from the lowest score.
    worst = level(min(hit, recall, faith, acc))
    status = {"good": "pass", "check": "check", "bad": "fail"}[worst]
    if status == "pass":
        return Diagnosis("pass")

    # The rules, from the most specific to the most general.
    if hit == 0:
        return Diagnosis(status, "Retrieval",
                         f"The expected paper ({row['source']}) is not among the retrieved chunks.")
    if recall < BAD and acc < GOOD:
        return Diagnosis(status, "Retrieval",
                         "The chunks miss facts the answer needs, so the LLM could not answer fully.")
    if recall < BAD:
        return Diagnosis(status, "Judge?",
                         "The answer is right, but the recall judge says the chunks lack the facts. "
                         "Check the chunks: the LLM may have used its own memory, or the judge is too strict.")
    if faith < BAD and acc >= GOOD:
        return Diagnosis(status, "Judge?",
                         "The answer is right and the chunks had the facts, but the faithfulness judge "
                         "found unsupported claims. Usually a judge error.")
    if acc < GOOD and faith < GOOD:
        return Diagnosis(status, "LLM",
                         "The answer doesn't match the reference and adds claims that are not in the chunks.")
    if acc < GOOD:
        return Diagnosis(status, "LLM or judge",
                         "The chunks had the facts and the answer stuck to them, but it doesn't match "
                         "the reference. Compare the two yourself.")
    return Diagnosis(status, "Minor", "Small deductions only. Read the answer to see if they matter.")
