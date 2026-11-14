"""Tests for the pure helpers in rag.caption (no model calls)."""

from rag.caption import CaptionCache, figure_number, mentions_figure, nearby_text


def el(type_: str, text: str, section: list[str], caption: str | None = None) -> dict:
    return {"type": type_, "text": text, "section_path": section, "caption": caption, "page": 1}


def test_figure_number():
    assert figure_number("Figure 5: The effect of position") == 5
    assert figure_number("Fig. 2. Multi-head attention") == 2
    assert figure_number("Table 1: Results") is None
    assert figure_number(None) is None


def test_mentions_figure_exact_number():
    assert mentions_figure("As Figure 5 shows, accuracy drops.", 5)
    assert mentions_figure("(see Fig. 5)", 5)
    assert mentions_figure("Figures 5 and 6 compare", 5)
    assert not mentions_figure("Figure 50 shows", 5)
    assert not mentions_figure("Figure 15 shows", 5)


def test_nearby_text_puts_mentions_first_then_nearest_same_section():
    a, b = ["T", "2 Method"], ["T", "3 Results"]
    elements = [
        el("paragraph", "far same-section", a),
        el("paragraph", "near same-section", a),
        el("figure", "Figure 3: Accuracy", a, caption="Figure 3: Accuracy"),
        el("paragraph", "other section, unrelated", b),
        el("paragraph", "Figure 3 shows a U-shape.", b),
    ]
    text = nearby_text(elements, 2, budget=1000)
    parts = text.split("\n\n")
    assert parts == ["Figure 3 shows a U-shape.", "near same-section", "far same-section"]


def test_nearby_text_respects_budget_and_empty_case():
    sec = ["T", "1 Intro"]
    elements = [el("paragraph", "x" * 100, sec), el("figure", "", sec)]
    assert len(nearby_text(elements, 1, budget=30)) == 30
    assert nearby_text([el("figure", "", sec)], 0) == "(none)"


def test_cache_key_changes_with_model_image_and_prompt():
    k = CaptionCache.key("m", b"img", "p")
    assert k == CaptionCache.key("m", b"img", "p")
    assert k != CaptionCache.key("m2", b"img", "p")
    assert k != CaptionCache.key("m", b"img2", "p")
    assert k != CaptionCache.key("m", b"img", "p2")
