"""Unit tests for the pure helpers in rag/parse.py (no Docling models needed)."""

from pathlib import Path

import pandas as pd

from rag.parse import (
    Element,
    SectionTracker,
    clean_text,
    doc_id_for,
    drop_repeated,
    heading_level,
    table_to_markdown,
)


def test_clean_text_fixes_ligatures_hyphens_and_spaces():
    assert clean_text("ﬁne-tuning") == "fine-tuning"
    assert clean_text("reim-\nbursement") == "reimbursement"
    assert clean_text("state-of-the-art") == "state-of-the-art"  # real hyphens stay
    assert clean_text("a   b\t c") == "a b c"


def test_heading_level_from_numbering():
    assert heading_level("2 Multi-Document QA") == 1
    assert heading_level("2.3 Results and Discussion") == 2
    assert heading_level("3.2.1 Details") == 3
    assert heading_level("A Ambiguity in Distractors") == 1
    assert heading_level("G.1 10 Total Retrieved Documents") == 2
    assert heading_level("References") == 1
    assert heading_level("Desired Answer") is None  # unknown: nested, not top level


def test_section_tracker_nests_unknown_headings():
    s = SectionTracker("Paper")
    s.enter(1, "2 Method")
    s.enter(2, "2.1 Setup")
    s.enter(None, "Desired Answer")
    assert s.path == ["Paper", "2 Method", "2.1 Setup", "Desired Answer"]
    s.enter(2, "2.2 Models")
    assert s.path == ["Paper", "2 Method", "2.2 Models"]
    s.enter(1, "3 Results")
    assert s.path == ["Paper", "3 Results"]


def test_table_to_markdown_is_compact_and_dedupes_merged_cells():
    long = "positional embedding instead of sinusoids"
    df = pd.DataFrame([["(E)", long, long, "4.92"]], columns=["", "N", "dmodel", "PPL"])
    md = table_to_markdown(df)
    assert md.splitlines()[0] == "|  | N | dmodel | PPL |"
    assert md.count(long) == 1
    assert "  |  " not in md.replace("|  |", "")  # no padding


def test_drop_repeated_removes_running_headers():
    els = [Element("paragraph", f"Page {p}", p) for p in range(1, 6)]
    els += [Element("paragraph", "Real content on page one.", 1)]
    kept = drop_repeated(els, n_pages=5)
    assert [e.text for e in kept] == ["Real content on page one."]


def test_doc_id_is_deterministic():
    assert doc_id_for(Path("data/raw/HR Leave Policy 2026.pdf")) == "hr-leave-policy-2026"
    assert doc_id_for(Path("data/raw/2307.03172.pdf")) == "2307.03172"
