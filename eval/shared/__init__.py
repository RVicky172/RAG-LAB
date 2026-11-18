"""
Code shared by every eval (RAG, single agent, multi-agent).

    judge.py       the LLM judge
    questions.py   read a question file, save the results
    status.py      score thresholds and the pass / check / fail result
    html_page.py   building blocks for the HTML reports (page, cards, bars, question rows)
    style.css      the look of every report (copied into the HTML, so each page is one file)
"""
