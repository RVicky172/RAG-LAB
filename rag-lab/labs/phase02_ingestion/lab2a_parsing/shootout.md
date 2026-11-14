# Parser shoot-out (Lab 2a)

Outputs: `data/lab_outputs/phase02_ingestion/lab2a_parsing/shootout/<parser>/<file>_pNNN.md`

Score 1 (unusable) to 5 (perfect) by comparing each output with the visual page. "n/a" = page has no such content.

| File | Page | Content | Parser | Reading order | Tables | Headings | Figures | Secs | Notes |
|------|------|---------|--------|---------------|--------|----------|---------|------|-------|
| 1706 | 9  | table  | pypdf   | 3 | 1 | 2 | n/a | 0.1 | Table flattened to number soup; rows (B) merged on one line; no header alignment; `dmodel = 1024on` missing space; page number `9` in text |
| 1706 | 9  | table  | pymupdf | 4 | 1 | 2 | n/a | 0.1 | Prose order fine; table still flat text with no row/column structure |
| 1706 | 9  | table  | docling | 5 | 4 | 4 | n/a | 10.5* | Real Markdown table with header + 20 rows; all numbers in the right columns. Defects: merged group labels `(A)`–`(E)` on one row only; `(E)` text copied into 8 cells; blank cells mean "same as base" (only explained in caption); `×106` should be ×10⁶ |
| 1706 | 13 | figure | pypdf   | 1 | n/a | 2 | 1 | 0.1 | Figure labels become ~70 lines of one-word word salad; image meaning lost |
| 1706 | 13 | figure | pymupdf | 3 | n/a | 2 | 2 | 0.0 | Rebuilds the figure's sentence + keeps the caption, but sentence duplicated (two panels), `<pad>` tokens and page number `13` in text; image meaning still lost |
| 1706 | 13 | figure | docling | 5 | n/a | 4 | 3 | 0.2 | Clean: caption + `<!-- image -->` placeholder, label noise dropped. Image content not read (no parser reads pixels): needs a vision-LLM caption |

| 2307 | 2  | 2-col prose | pypdf   | 4 | n/a | 2 | n/a | 0.1 | Columns in the right order (this PDF stores text column by column), but 27 line-end hyphens (`state-of-` / `the-art`) and hard line breaks |
| 2307 | 2  | 2-col prose | pymupdf | 1 | n/a | 2 | n/a | 0.1 | **Columns interleaved line by line**: `sort=True` sorts by y across the whole page, so each line mixes left and right column text. Unusable |
| 2307 | 2  | 2-col prose | docling | 5 | n/a | 4 | n/a | 6.8* | Correct column order, paragraphs rejoined, hyphenation fixed. Glitch: real hyphens dropped too (`state-ofthe-art`, `Claude1.3`, `Ushaped`) |
| 2307 | 5  | figure + table | pypdf   | 3 | 1 | 2 | 1 | 0.0 | Table 1 flattened; Figure 5 axis labels mixed into text |
| 2307 | 5  | figure + table | pymupdf | 1 | 1 | 2 | 1 | 0.0 | Chart axes printed as text art (`Accuracy6560`); Table 1 rows interleaved with prose from the other column |
| 2307 | 5  | figure + table | docling | 5 | 5 | 5 | 3 | 0.7 | Table 1 as a perfect Markdown table; Figure 5 = caption + `<!-- image -->`; heading `## 2.3 Results and Discussion` detected |

\* Includes loading Docling's models; page 13 (0.2s) is closer to the warm per-page speed.

## Key observations

- **pypdf** loses the context of both the table and the figure completely. Unusable for this corpus.
- **PyMuPDF** reads prose and figure text better than pypdf, but tables are still flat text and figures are still just their labels.
- **Docling** is the only parser that keeps table structure and separates figures from prose. Its gaps are known and fixable:
  1. Merged-cell labels and "same as base" blanks: keep the **table caption in the same chunk** as the table.
  2. Figures: send the image to the **vision LLM** for a caption (Step 5).
- Docling's default Markdown pads table cells with spaces (11,728 chars on p.9). `compact_tables=True` cut it to 4,068 chars with the same content, so fewer wasted embedding tokens.
- OCR is off (`do_ocr=False`): the PDFs are digital and have a text layer.

## Multi-column test (added 2307.03172 "Lost in the Middle" and 2309.15217 "RAGAS", both two-column)

- **PyMuPDF `sort=True` is the worst parser on two-column pages**: it sorts text by vertical position across the whole page, so left and right columns are interleaved line by line. Every chunk would be a mix of two unrelated passages.
- **pypdf got the column order right here only by luck**: this PDF happens to store its text column by column. Other PDFs (Word exports, scanned + OCR) often don't. Not something to rely on.
- **Docling uses a layout model** that sees the columns, so the reading order is correct by design. It also rejoins paragraphs and fixes line-end hyphenation.
- New Docling defect: over-eager de-hyphenation drops real hyphens (`state-ofthe-art`, `multidocument`, `Claude1.3`, `Ushaped`). Matters for exact-match/BM25 on model names and IDs.

## Decision

Parser for my corpus: **Docling** (OCR off, table structure on, compact tables).
Why: the corpus is PDF with tables and multi-column layout; only Docling keeps table structure and reading order on two-column pages (PyMuPDF interleaves columns, pypdf only works when the PDF happens to store text in column order), and it isolates figures for captioning. pypdf stays as a fast fallback for single-column prose only.
Cost (seconds per page × pages in corpus): measured 0.2s (figure page) to 10.5s (table page, including model load). Time one full document after warm-up to get a real s/page, then multiply by the corpus page count.
