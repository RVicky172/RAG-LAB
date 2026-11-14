"""Phase 02, Lab 2a: parse the same pages with pypdf, PyMuPDF and Docling.

Saves each parser's output per page so you can compare them side by side,
then score them in shootout.md (reading order, tables, headings, figures).

Usage (from rag-lab/):
    uv run python labs/phase02_ingestion/lab2a_parsing/parser_shootout.py data/raw/1706.03762v7.pdf 9 13
    uv run python labs/phase02_ingestion/lab2a_parsing/parser_shootout.py data/raw/1706.03762v7.pdf 9 --parsers docling

Output:
    data/lab_outputs/phase02_ingestion/lab2a_parsing/shootout/<parser>/<pdf-stem>_p009.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag.paths import lab_output_dir

OUT_DIR = lab_output_dir("phase02_ingestion", "lab2a_parsing") / "shootout"


@dataclass
class PageResult:
    parser: str
    page: int
    text: str
    seconds: float
    elements: Counter[str] = field(default_factory=Counter)


# ---------- parsers: each returns (text, element counts) for one 1-based page ----------

def parse_pypdf(path: Path, page: int) -> tuple[str, Counter[str]]:
    """Text layer only, in content-stream order (what Phase 01 used)."""
    from pypdf import PdfReader

    return PdfReader(path).pages[page - 1].extract_text() or "", Counter()


def parse_pymupdf(path: Path, page: int) -> tuple[str, Counter[str]]:
    """Text layer, with blocks sorted top-to-bottom, left-to-right."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return doc[page - 1].get_text(sort=True), Counter()


class DoclingParser:
    """Layout model + table-structure model. Loads models once, reuses them."""

    def __init__(self) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Digital PDFs already have a text layer: OCR only adds time.
        # Set do_ocr=True for scanned documents.
        options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def __call__(self, path: Path, page: int) -> tuple[str, Counter[str]]:
        doc = self.converter.convert(path, page_range=(page, page)).document
        counts = Counter(str(item.label) for item, _ in doc.iterate_items(page_no=page))
        # compact_tables: no space padding in cells (padding wastes embedding tokens)
        return doc.export_to_markdown(page_no=page, compact_tables=True), counts


PARSERS: dict[str, Callable[[], Callable[[Path, int], tuple[str, Counter[str]]]]] = {
    "pypdf": lambda: parse_pypdf,
    "pymupdf": lambda: parse_pymupdf,
    "docling": DoclingParser,
}


# ---------- run, save, report ----------

def run(path: Path, pages: list[int], parser_names: list[str]) -> list[PageResult]:
    results: list[PageResult] = []
    for name in parser_names:
        print(f"[{name}] loading...", flush=True)
        parse = PARSERS[name]()
        for page in pages:
            start = time.perf_counter()
            text, elements = parse(path, page)
            seconds = time.perf_counter() - start
            results.append(PageResult(name, page, text, seconds, elements))
            print(f"[{name}] page {page}: {seconds:.1f}s", flush=True)
    return results


def save(path: Path, result: PageResult) -> Path:
    out = OUT_DIR / result.parser / f"{path.stem}_p{result.page:03d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.text, encoding="utf-8")
    return out


def print_summary(results: list[PageResult]) -> None:
    print(f"\n{'parser':<8} {'page':>4} {'secs':>6} {'chars':>6}  elements (Docling only)")
    for r in results:
        elements = ", ".join(f"{k} {v}" for k, v in r.elements.most_common()) or "-"
        print(f"{r.parser:<8} {r.page:>4} {r.seconds:>6.1f} {len(r.text):>6}  {elements}")
    print(f"\nOpen the files in {OUT_DIR}/ side by side with the PDF and score them in shootout.md.")
    print("Note: Docling's first page includes model loading time.")


def parse_pages(specs: list[str]) -> list[int]:
    """Turn ["9", "13-14"] into [9, 13, 14]."""
    pages: set[int] = set()
    for spec in specs:
        start, _, end = spec.partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return sorted(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PDF parsers on the same pages.")
    parser.add_argument("pdf", type=Path, help="path to a PDF")
    parser.add_argument("pages", nargs="+", help="pages to parse, e.g. 9 13 or 8-9")
    parser.add_argument("--parsers", nargs="+", choices=PARSERS, default=list(PARSERS))
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"Not a file: {args.pdf}")

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    logging.basicConfig(level=logging.WARNING)  # keep Docling's INFO logs quiet

    results = run(args.pdf, parse_pages(args.pages), args.parsers)
    for result in results:
        save(args.pdf, result)
    print_summary(results)


if __name__ == "__main__":
    main()
