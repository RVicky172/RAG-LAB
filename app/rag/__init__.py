"""
RAG (Doc 1): answer questions from the papers in two stages.

    ingest.py   once:          PDFs -> chunks -> vectors in Qdrant   (python -m app.rag.ingest)
    query.py    per question:  retrieve -> augment -> generate       (python -m app.rag.query "...")
"""
