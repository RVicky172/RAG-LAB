"""
Doc 1: evaluate the RAG pipeline.

    questions.jsonl  the test set: question, reference answer, expected paper
    evaluate.py      run the eval                          (python -m eval.rag.evaluate)
    tokens.py        split the answer call's tokens into instructions / context / question / output
    diagnose.py      pass / check / fail per question, and the likely cause
    report.py        write results/report.html
    results/         output of the last run: results.json + report.html
"""
