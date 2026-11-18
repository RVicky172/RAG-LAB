"""
Ingestion: load the PDFs into the vector database.

Steps:
    1. Read every page of every PDF in `data/raw/`.
    2. Split each page into small chunks of text.
    3. Store the chunks in Qdrant. qdrant-client turns each chunk into a vector
       (an "embedding") for us, so we never call the embedding model ourselves.

Why chunks? An LLM can only read a limited amount of text, and a small chunk
about one topic is easier to find with search than a whole page.

Run: uv run python -m app.rag.ingest
"""

from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from app.shared import config


def load_pages(folder: str = "data/raw"):
    """Read every page of every PDF in a folder.

    Args:
        folder: Folder that holds the PDF files.

    Yields:
        One dict per page: `{"source": file name, "page": page number
        (starting at 1), "text": the page's text}`. The source and page are
        kept so the answer can cite where it found something.
    """
    for pdf in sorted(Path(folder).glob("*.pdf")):
        for page_no, page in enumerate(pymupdf.open(pdf), start=1):
            yield {"source": pdf.name, "page": page_no, "text": page.get_text()}


def split_into_chunks(pages) -> list[dict]:
    """Split each page into small, overlapping chunks.

    Args:
        pages: The page dicts from `load_pages()`.

    Returns:
        A list of chunk dicts. Each one has the same keys as a page
        (source, page, text), but `text` holds only the chunk.
    """
    # The splitter tries to cut at paragraphs first, then lines, then words,
    # so a chunk rarely stops in the middle of a sentence.
    # chunk_size: the most characters in one chunk.
    # chunk_overlap: neighbouring chunks share this many characters, so a
    # sentence that sits on a cut still appears whole in one of them.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    chunks = []
    for page in pages:
        for text in splitter.split_text(page["text"]):
            chunks.append({**page, "text": text})  # copy source/page, replace the text
    return chunks


def main():
    """Rebuild the Qdrant collection from the PDFs."""
    chunks = split_into_chunks(load_pages())
    print(f"{len(chunks)} chunks")

    client = QdrantClient(config.QDRANT_URL)

    # Start fresh every time, so old chunks from an earlier run never mix in.
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)

    # A collection must know the vector size up front (384 for bge-small).
    # COSINE means "how similar are these two vectors", the usual choice for text.
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(config.EMBED_MODEL),
            distance=models.Distance.COSINE,
        ),
    )

    # Each chunk becomes one "point":
    #   id      -> a unique number
    #   payload -> the data we get back from a search (source, page, text)
    #   vector  -> models.Document asks qdrant-client to embed the text for us
    points = [
        models.PointStruct(
            id=i,
            payload=chunk,
            vector=models.Document(text=chunk["text"], model=config.EMBED_MODEL),
        )
        for i, chunk in enumerate(chunks)
    ]

    # batch_size: send 64 points per request instead of everything at once.
    # wait=True: wait until Qdrant has saved them, so the count below is correct.
    client.upload_points(config.COLLECTION, points=points, batch_size=64, wait=True)
    print("stored:", client.count(config.COLLECTION).count)


if __name__ == "__main__":
    main()
