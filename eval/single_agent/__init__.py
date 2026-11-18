"""
Doc 2: evaluate the single agent. The multi-agent eval (Doc 3) reuses this code.

    questions.jsonl  the test set: question, reference goal, expected tools
    evaluate.py      run the eval; main() also runs the multi-agent eval   (python -m eval.single_agent.evaluate)
    diagnose.py      pass / check / fail per question, and the likely cause
    report.py        write results/report.html (for the multi-agent eval too)
    results/         output of the last run: results.json + report.html
"""
