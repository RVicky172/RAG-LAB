"""rag/parse.py: PDF -> typed elements with page numbers and section paths.

Phase 02 milestone. Decision (labs/phase02_ingestion/lab2a_parsing/shootout.md): Docling for this corpus
(two-column papers with tables and figures). pypdf is kept as a baseline.

Output per document (data/parsed/<parser>/<doc_id>.json; figures in data/figures/<doc_id>/):
    {"doc_id", "title", "source_uri", "content_hash", "parser", "n_pages",
     "elements": [{"type", "text", "page", "section_path", ...}, ...]}

Element types: heading, paragraph, list_item, table, figure, caption,
footnote, formula, code.

Usage (from rag-lab/):
    uv run python -m rag.parse data/raw            # every PDF in the folder
    uv run python -m rag.parse data/raw/2307.03172.pdf --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from rag.paths import FIGURES, parsed_dir


# ---------- data model ----------

@dataclass
class Element:
    type: str                        # heading | paragraph | list_item | table | figure | caption | ...
    text: str                        # cleaned text; tables: caption + Markdown; figures: caption
    page: int                        # 1-based, for citations
    section_path: list[str] = field(default_factory=list)
    level: int | None = None         # headings only
    markdown: str | None = None      # tables only: the raw Markdown table
    caption: str | None = None       # tables and figures
    image_path: str | None = None    # figures only: PNG for vision captioning (Step 5)


class Parser(Protocol):
    name: str

    def parse(self, path: Path) -> dict: ...


# ---------- cleaning (applies to every parser's text) ----------

HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*([a-z])")  # "reim-\nbursement" -> "reimbursement"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ligatures: "ﬁ" -> "fi"; full-width chars
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_repeated(elements: list[Element], n_pages: int, share: float = 0.5) -> list[Element]:
    """Remove short lines that recur on most pages (running headers/footers, page numbers).

    Docling already moves most of these to a separate layer; this is a safety net
    and does the real work for the pypdf baseline.
    """
    if n_pages < 3:
        return elements

    def key(e: Element) -> str:
        return re.sub(r"\d+", "#", e.text).strip().lower()

    pages_with = Counter()
    for k in {(key(e), e.page) for e in elements if len(e.text) < 80}:
        pages_with[k[0]] += 1
    repeated = {k for k, n in pages_with.items() if n / n_pages >= share}
    return [e for e in elements if not (len(e.text) < 80 and key(e) in repeated)]


# ---------- tables ----------

def table_to_markdown(df) -> str:
    """Compact Markdown table (no space padding: padding wastes embedding tokens).

    A merged cell spanning several columns comes out of Docling copied into each
    column ("positional embedding instead of sinusoids" x 8); keep it once per row.
    """
    if df.empty:
        return ""

    def cell(value) -> str:
        return clean_text(str(value)).replace("|", "\\|").replace("\n", " ")

    def dedupe(cells: list[str]) -> list[str]:
        return [c if i == 0 or c != cells[i - 1] or len(c) < 12 else "" for i, c in enumerate(cells)]

    header = dedupe([cell(c) for c in df.columns])
    rows = [dedupe([cell(v) for v in row]) for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---------- section paths ----------

NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")      # "2.3 Results" -> "2.3"
APPENDIX = re.compile(r"^([A-Z](?:\.\d+)*)\.?\s+\S")    # "A Details", "G.1 10 Docs" -> "A", "G.1"
TOP_LEVEL = re.compile(
    r"^(abstract|introduction|conclusions?|acknowledge?ments?|references|bibliography|"
    r"appendix|appendices|related work|limitations|ethics statement|broader impact)\b",
    re.IGNORECASE,
)


def heading_level(text: str) -> int | None:
    """Docling marks most paper headings as level 1; recover depth from numbering.

    Returns None for an unnumbered, non-standard heading (often text inside a
    figure that Docling mislabelled, e.g. "Desired Answer").
    """
    if m := NUMBERED.match(text) or APPENDIX.match(text):
        return m.group(1).count(".") + 1
    if TOP_LEVEL.match(text):
        return 1
    return None


class SectionTracker:
    """Keeps the current heading stack, e.g. ["Lost in the Middle", "2 Multi-Doc QA", "2.3 Results"]."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.stack: list[tuple[int, str]] = []

    def enter(self, level: int | None, text: str) -> int:
        """Push a heading; unknown level = one below the current section (never resets the path)."""
        if level is None:
            level = (self.stack[-1][0] + 1) if self.stack else 1
        self.stack = [(l, t) for l, t in self.stack if l < level] + [(level, text)]
        return level

    @property
    def path(self) -> list[str]:
        return [self.title] + [t for _, t in self.stack]


# ---------- document-level helpers ----------

def doc_id_for(path: Path) -> str:
    """Deterministic id from the file name: re-runs overwrite, never duplicate."""
    return re.sub(r"[^a-z0-9.]+", "-", path.stem.lower()).strip("-")


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def make_doc(path: Path, parser: str, title: str, n_pages: int, elements: list[Element]) -> dict:
    return {
        "doc_id": doc_id_for(path),
        "title": title,
        "source_uri": path.as_posix(),
        "content_hash": file_hash(path),
        "parser": parser,
        "n_pages": n_pages,
        "elements": [asdict(e) for e in elements],
    }


# ---------- Docling (main parser) ----------

class DoclingParser:
    """Layout model (reading order, headings) + table-structure model. OCR off: digital PDFs."""

    name = "docling"

    # Docling label -> our element type. Anything else (page_header, page_footer, ...) is dropped.
    TYPE_MAP = {
        "title": "heading",
        "section_header": "heading",
        "text": "paragraph",
        "paragraph": "paragraph",
        "reference": "paragraph",
        "list_item": "list_item",
        "caption": "caption",
        "footnote": "footnote",
        "formula": "formula",
        "code": "code",
    }

    def __init__(self, save_figures: bool = True) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        self.save_figures = save_figures
        options = PdfPipelineOptions(
            do_ocr=False,                         # set True for scanned PDFs
            do_table_structure=True,
            generate_picture_images=save_figures,  # crop figures for the vision LLM
            images_scale=2.0,                      # ~144 DPI crops, readable axis labels
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse(self, path: Path) -> dict:
        doc = self.converter.convert(path).document
        doc_id = doc_id_for(path)
        items = [item for item, _ in doc.iterate_items()]

        # Captions attached to a table/figure are emitted with it, not again on their own.
        attached = {ref.cref for item in items for ref in getattr(item, "captions", [])}
        title = self._title(items) or path.stem
        sections = SectionTracker(title)
        elements: list[Element] = []
        fig_count = Counter()

        for item in items:
            label = str(item.label)
            page = item.prov[0].page_no if item.prov else 0

            if label == "table":
                caption = clean_text(item.caption_text(doc))
                table_md = table_to_markdown(item.export_to_dataframe(doc=doc))
                if not table_md:
                    continue
                # The caption travels with the table: blank cells / units are explained there.
                text = f"{caption}\n\n{table_md}" if caption else table_md
                elements.append(Element("table", text, page, sections.path,
                                        markdown=table_md, caption=caption or None))

            elif label == "picture":
                caption = clean_text(item.caption_text(doc))
                fig_count[page] += 1
                image_path = self._save_image(item, doc, doc_id, page, fig_count[page])
                elements.append(Element("figure", caption, page, sections.path,
                                        caption=caption or None, image_path=image_path))

            elif item.self_ref in attached:
                continue

            elif (etype := self.TYPE_MAP.get(label)) is not None:
                text = clean_text(item.text)
                if not text:
                    continue
                if etype == "heading":
                    if text == title:
                        continue  # the title is the root of every section path
                    level = sections.enter(heading_level(text), text)
                    elements.append(Element("heading", text, page, sections.path, level=level))
                else:
                    elements.append(Element(etype, text, page, sections.path))

        n_pages = len(doc.pages)
        return make_doc(path, self.name, title, n_pages, drop_repeated(elements, n_pages))

    @staticmethod
    def _title(items: list) -> str | None:
        """Explicit title, else the first heading on page 1 (papers rarely get a 'title' label)."""
        for item in items:
            if str(item.label) == "title":
                return clean_text(item.text)
        for item in items:
            if str(item.label) == "section_header" and item.prov and item.prov[0].page_no == 1:
                return clean_text(item.text)
        return None

    def _save_image(self, item, doc, doc_id: str, page: int, n: int) -> str | None:
        if not self.save_figures or (image := item.get_image(doc)) is None:
            return None
        out = FIGURES / doc_id / f"p{page:03d}-fig{n}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        return out.as_posix()


# ---------- pypdf (baseline, for ablations) ----------

class PypdfParser:
    """Phase 01 behaviour: text layer only, one paragraph element per blank-line block."""

    name = "pypdf"

    def parse(self, path: Path) -> dict:
        from pypdf import PdfReader

        reader = PdfReader(path)
        elements = [
            Element("paragraph", clean_text(block), page)
            for page, pdf_page in enumerate(reader.pages, start=1)
            for block in re.split(r"\n\s*\n", pdf_page.extract_text() or "")
            if block.strip()
        ]
        n_pages = len(reader.pages)
        return make_doc(path, self.name, path.stem, n_pages, drop_repeated(elements, n_pages))


PARSERS: dict[str, type] = {"docling": DoclingParser, "pypdf": PypdfParser}


# ---------- run over a corpus ----------

def output_path(path: Path, parser: str) -> Path:
    """data/parsed/<parser>/<doc_id>.json"""
    return parsed_dir(parser) / f"{doc_id_for(path)}.json"


def is_up_to_date(path: Path, out: Path) -> bool:
    """Skip unchanged files: same content hash as the last parse."""
    if not out.exists():
        return False
    return json.loads(out.read_text(encoding="utf-8")).get("content_hash") == file_hash(path)


def summarize(doc: dict) -> str:
    counts = Counter(e["type"] for e in doc["elements"])
    return ", ".join(f"{k} {v}" for k, v in counts.most_common())


def parse_corpus(paths: list[Path], parser_name: str = "docling", force: bool = False) -> list[Path]:
    parser: Parser | None = None  # load models only if something needs parsing
    written: list[Path] = []
    for path in paths:
        out = output_path(path, parser_name)
        if not force and is_up_to_date(path, out):
            print(f"skip   {path.name} (unchanged)")
            continue
        parser = parser or PARSERS[parser_name]()
        start = time.perf_counter()
        doc = parser.parse(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        secs = time.perf_counter() - start
        print(f"parsed {path.name}: {doc['n_pages']} pages in {secs:.1f}s "
              f"({secs / doc['n_pages']:.2f}s/page) -> {out}\n       {summarize(doc)}")
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDFs into typed elements.")
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF files or folders")
    ap.add_argument("--parser", choices=PARSERS, default="docling")
    ap.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)

    paths = sorted(
        p for i in args.inputs
        for p in ([i] if i.is_file() else i.glob("*.pdf"))
        if p.suffix.lower() == ".pdf"
    )
    if not paths:
        raise SystemExit("No PDFs found.")
    parse_corpus(paths, args.parser, args.force)


if __name__ == "__main__":
    main()
