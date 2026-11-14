"""Tests for rag/chunk.py. Uses the embedding tokenizer (downloads tokenizer files once)."""

from rag.chunk import (
    RecursiveChunker,
    Sentence,
    StructureChunker,
    count_tokens,
    merge_spans,
    pack_sentences,
    split_sentences,
    split_spans,
    split_table,
)


def test_split_sentences_keeps_abbreviations():
    text = "RAG works well, e.g. on QA. Lewis et al. showed this. (Results) are in Table 1."
    assert split_sentences(text) == [
        "RAG works well, e.g. on QA.",
        "Lewis et al. showed this.",
        "(Results) are in Table 1.",
    ]


def test_pack_sentences_caps_group_size():
    sents = [Sentence(f"s{i}", 1, ["Doc"], 30) for i in range(10)]
    groups = pack_sentences(sents, 100)
    assert all(sum(s.n_tokens for s in g) <= 100 for g in groups)
    assert sum(len(g) for g in groups) == 10


def make_doc(elements):
    return {"doc_id": "doc", "title": "Doc", "source_uri": "data/raw/doc.pdf", "elements": elements}


def para(text, page=1, path=("Doc", "1 Intro")):
    return {"type": "paragraph", "text": text, "page": page, "section_path": list(path)}


def test_split_spans_respects_limit_and_covers_text():
    text = "\n\n".join(f"Sentence number {i} is here. Another one follows." for i in range(60))
    spans = split_spans(text, 0, len(text), 40)
    assert all(count_tokens(text[a:b]) <= 40 for a, b in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)


def test_merge_spans_overlaps():
    text = " ".join(f"w{i}" for i in range(400))
    spans = split_spans(text, 0, len(text), 5)
    merged = merge_spans(text, spans, 100, 20)
    assert len(merged) > 1
    assert merged[1][0] < merged[0][1]  # second chunk starts inside the first: overlap


def test_table_split_repeats_caption_and_header():
    header = "| Model | Score |\n|---|---|"
    rows = "\n".join(f"| model-{i} | {i}.0 |" for i in range(200))
    markdown = f"{header}\n{rows}"
    table = {"text": f"Table 1: Scores.\n\n{markdown}", "markdown": markdown, "caption": "Table 1: Scores."}
    pieces = split_table(table, 120)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("Table 1: Scores.")
        assert "| Model | Score |\n|---|---|" in piece
        assert count_tokens(piece) <= 120
    assert sum(p.count("| model-") for p in pieces) == 200  # no row lost or duplicated


def test_structure_chunker_prepends_section_and_isolates_tables():
    doc = make_doc([
        {"type": "heading", "text": "1 Intro", "page": 1, "section_path": ["Doc", "1 Intro"]},
        para("Intro text."),
        {"type": "table", "text": "| a |\n|---|\n| 1 |", "markdown": "| a |\n|---|\n| 1 |",
         "caption": None, "page": 2, "section_path": ["Doc", "1 Intro"]},
        para("Method text.", page=3, path=("Doc", "2 Method")),
    ])
    chunks = StructureChunker(200, 20).chunk(doc)
    assert [c.element_type for c in chunks] == ["text", "table", "text"]
    assert chunks[0].text.startswith("Doc > 1 Intro\n")
    assert chunks[2].section_path == "Doc > 2 Method"
    assert chunks[1].page == 2


def test_chunk_ids_are_deterministic():
    doc = make_doc([para("Some text. " * 200, page=p) for p in (1, 2)])
    first = [c.chunk_id for c in RecursiveChunker(100, 10).chunk(doc)]
    second = [c.chunk_id for c in RecursiveChunker(100, 10).chunk(doc)]
    assert first == second
    assert first[0] == "doc#v1#p1#c000"
