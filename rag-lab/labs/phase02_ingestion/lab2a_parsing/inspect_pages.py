"""Phase 02, Step 1: see what pypdf really extracts from a PDF.

Prints the raw text layer of selected pages, with line numbers, plus
automatic hints for common parsing defects. Compare each page with the
visual PDF and write what you find in defects.md.

Usage (from rag-lab/):
    # 1. Overview: find the tricky pages (one row per page)
    uv run python labs/phase02_ingestion/lab2a_parsing/inspect_pages.py data/raw/2005.11401v4.pdf

    # 2. Inspect chosen pages (single numbers or ranges, 1-based)
    uv run python labs/phase02_ingestion/lab2a_parsing/inspect_pages.py data/raw/2005.11401v4.pdf 3 5-7

    # 3. Also save each page's text for the parser shoot-out (Lab 2a)
    uv run python labs/phase02_ingestion/lab2a_parsing/inspect_pages.py data/raw/2005.11401v4.pdf 3 5-7 --save
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from rag.paths import lab_output_dir

OUT_DIR = lab_output_dir("phase02_ingestion", "lab2a_parsing") / "inspect" / "pypdf"

LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"
HYPHEN_AT_EOL = re.compile(r"[A-Za-z]-$")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")

# A line that recurs on at least this share of pages is likely a header/footer.
REPEAT_SHARE = 0.5
# A line with this many numbers and few words is likely a flattened table row.
TABLE_MIN_NUMBERS = 3


@dataclass
class PageReport:
    page: int
    text: str
    hints: list[str]

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


def parse_pages(specs: list[str], n_pages: int) -> list[int]:
    """Turn ["3", "5-7"] into [3, 5, 6, 7], checking bounds."""
    pages: list[int] = []
    for spec in specs:
        start, _, end = spec.partition("-")
        first, last = int(start), int(end or start)
        if not 1 <= first <= last <= n_pages:
            raise SystemExit(f"Page spec '{spec}' is outside 1-{n_pages}.")
        pages.extend(range(first, last + 1))
    return sorted(set(pages))


def normalize(line: str) -> str:
    """Key for header/footer detection: ignore digits (page numbers) and spacing."""
    return re.sub(r"\d+", "#", line).strip().lower()


def find_repeated_lines(texts: list[str]) -> set[str]:
    """Lines that appear on many pages: candidate headers and footers."""
    if len(texts) < 3:
        return set()
    counts = Counter(
        key for text in texts for key in {normalize(l) for l in text.splitlines()} if key
    )
    return {key for key, n in counts.items() if n / len(texts) >= REPEAT_SHARE}


def detect_defects(text: str, repeated: set[str]) -> list[str]:
    """Heuristic hints. They point you at problems; your eyes confirm them."""
    lines = [l for l in text.splitlines() if l.strip()]
    hints: list[str] = []

    if not lines:
        return ["NO TEXT LAYER: scanned page? needs OCR"]

    if ligatures := sorted({c for c in text if c in LIGATURES}):
        hints.append(f"ligatures {''.join(ligatures)}: normalize Unicode (NFKC)")

    if hyphenated := sum(bool(HYPHEN_AT_EOL.search(l.rstrip())) for l in lines):
        hints.append(f"{hyphenated} line(s) end in a hyphen: words split across lines")

    if headers := [l.strip() for l in lines if normalize(l) in repeated]:
        hints.append(f"header/footer candidates: {headers[:3]}")

    table_rows = [
        l for l in lines
        if len(NUMBER.findall(l)) >= TABLE_MIN_NUMBERS
        and len(re.findall(r"[A-Za-z]{3,}", l)) <= len(NUMBER.findall(l))
    ]
    if table_rows:
        hints.append(f"{len(table_rows)} number-heavy line(s): flattened table? e.g. {table_rows[0].strip()[:60]!r}")

    short = sum(len(l.strip()) < 25 for l in lines)
    if short / len(lines) > 0.4:
        hints.append(f"{short}/{len(lines)} very short lines: columns, table cells or figure labels")

    return hints


def build_reports(reader: PdfReader) -> list[PageReport]:
    texts = [page.extract_text() or "" for page in reader.pages]
    repeated = find_repeated_lines(texts)
    return [
        PageReport(page=i, text=text, hints=detect_defects(text, repeated))
        for i, text in enumerate(texts, start=1)
    ]


def print_overview(path: Path, reports: list[PageReport]) -> None:
    print(f"{path.name}: {len(reports)} pages\n")
    print(f"{'page':>4}  {'chars':>6}  {'lines':>5}  hints")
    for r in reports:
        flags = "; ".join(h.split(":")[0] for h in r.hints) or "-"
        print(f"{r.page:>4}  {len(r.text):>6}  {len(r.lines):>5}  {flags}")
    print("\nPick ~5 pages with hints (multi-column, tables, figures) and rerun with page numbers.")


def print_page(path: Path, report: PageReport) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{path.name}  page {report.page}  ({len(report.text)} chars)\n{bar}")
    width = len(str(len(report.lines)))
    for n, line in enumerate(report.lines, start=1):
        print(f"{n:>{width}} | {line}")
    print("-" * 72)
    for hint in report.hints or ["no automatic hints: still compare with the visual page"]:
        print(f"  ! {hint}")


def save_page(path: Path, report: PageReport) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_p{report.page:03d}.txt"
    out.write_text(report.text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pypdf text extraction page by page.")
    parser.add_argument("pdf", type=Path, help="path to a PDF")
    parser.add_argument("pages", nargs="*", help="pages to print, e.g. 3 5-7 (omit for overview)")
    parser.add_argument("--save", action="store_true", help=f"save page text to {OUT_DIR}/")
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"Not a file: {args.pdf}")

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    reports = build_reports(PdfReader(args.pdf))

    if not args.pages:
        print_overview(args.pdf, reports)
        return

    for page in parse_pages(args.pages, len(reports)):
        report = reports[page - 1]
        print_page(args.pdf, report)
        if args.save:
            print(f"  saved -> {save_page(args.pdf, report)}")


if __name__ == "__main__":
    main()
