# Parsing defects (Phase 02, Step 1)

**Parsers checked:** pypdf (baseline, used in Phase 01), then PyMuPDF and Docling (Lab 2a, see [shootout.md](shootout.md))
**Corpus types:** digital PDF, multi-column, some tables, few figures. No scans, HTML, Markdown or DOCX.

## Pages inspected

| # | File | Page | Why it's tricky (what you see visually) |
|---|------|------|-----------------------------------------|
| 1 | 1706.03762v7.pdf | 9 | Table 3 (grouped rows A–E, merged cells) + prose |
| 2 | 1706.03762v7.pdf | 13 | Attention-visualization figure (words linked by lines) |
| 3 | 1706.03762v7.pdf | 1 | Title page: authors, affiliations, emails (many short lines) |
| 4 | 1706.03762v7.pdf | 3 | Figure 1 (architecture diagram) taking half the page |
| 5 | 2307.03172.pdf | 2 | Two-column prose (Lost in the Middle) |
| 6 | 2307.03172.pdf | 5 | Two-column: Figure 5 (3 line charts) + Table 1 + prose |

## Defects found: pypdf

| Defect | Example (file p.N, line) | Impact on retrieval | Fix (parser / cleaning step) |
|--------|--------------------------|---------------------|------------------------------|
| Table flattened into number soup | 1706 p.9 l.12: `(B) 16 5.16 25.1 58 32 5.01 25.4 60`, two rows merged, no header | Numbers lose their column meaning; questions about the table can't be answered | Layout-aware parser (Docling), table as Markdown |
| Figure turned into word salad | 1706 p.13 l.3–70: `It / is / in / this / spirit ...` one word per line | Meaningless chunk that can still be retrieved; the figure's meaning is lost | Detect figure regions, drop the label text, caption the image with the vision LLM |
| Missing space between tokens | 1706 p.9 l.44: `dmodel = 1024on` | BM25 / exact-match lookups miss terms | Layout-aware parser; check after Docling |
| Ligatures | `ﬁ`, `ﬂ` on almost every page of 2005.11401 | `ﬁne-tuning` ≠ `fine-tuning` for BM25 | Unicode NFKC normalization |
| Hyphenation at line end | 2005.11401: 1–10 per page (overview) | Split words never match | Join `-\n` + lowercase letter |
| Page numbers / headers in text | 1706 p.9 l.52: `9` | Noise inside chunks | Drop repeated lines (header/footer detection) |
| Column reading order | 2307 p.2: correct here, but only because this PDF stores text column by column | Would mix two passages on PDFs stored differently | Don't rely on it: layout-aware parser |
| Narrow-column hyphenation | 2307: 10–29 line-end hyphens per page (vs 1–4 in single-column papers) | Many split words | Join `-\n` + lowercase letter (or Docling) |

## Defects found: PyMuPDF

| Defect | Example | Impact on retrieval | Fix |
|--------|---------|---------------------|-----|
| Table still flat text | 1706 p.9: no row/column structure | Same as pypdf | Docling |
| Figure text duplicated + junk tokens | 1706 p.13: sentence repeated for both panels, `<pad>` tokens, page number `13` | Noisy chunk; image meaning still missing | Docling + vision caption |
| **Columns interleaved line by line** | 2307 p.2: `...inter alia) have   document question task is lower than its perfor-` (left + right column on one line) | Every chunk mixes two unrelated passages: worst possible input | Docling (layout model) |
| Chart axes printed as text art | 2307 p.5: `Accuracy6560`, rows of `75   75   75` | Noise chunk | Docling + vision caption |

## Defects found: Docling (remaining after switching)

| Defect | Example | Impact on retrieval | Fix |
|--------|---------|---------------------|-----|
| Image content not read | 1706 p.13: only caption + `<!-- image -->` | Questions about what the figure shows can't be answered | Vision-LLM caption (Step 5, `caption_figure()`) |
| Merged-cell label on one row only | 1706 p.9: `(A)` only on the h=16 row; h=1, 4, 32 rows have an empty label | A row can't be tied to its experiment group | Forward-fill the group column; keep caption with table |
| Merged cell copied into every column | 1706 p.9: `(E)` "positional embedding instead of sinusoids" × 8 | Noise in the chunk | Collapse repeated identical cells in a row |
| Blank cells mean "same as base" | 1706 p.9: h=1 row has empty N, dmodel... | Meaning only in the caption | Always keep the **caption in the same chunk** as the table |
| Superscript lost | 1706 p.9: `params ×106` (means ×10⁶) | Misread numbers | Minor; note it, keep the caption |
| Real hyphens dropped | 2307 p.2/p.5: `state-ofthe-art`, `multidocument`, `Claude1.3`, `Ushaped` | Exact match / BM25 on model names and IDs fails (`Claude-1.3` ≠ `Claude1.3`) | Watch in eval; post-process or keep the raw text layer for IDs |
| Table padding (fixed) | p.9 was 11,728 chars | Wasted embedding tokens | `compact_tables=True` → 4,068 chars |

## Decision

Based on the defects above, the parser for this corpus is: **Docling** (OCR off, table structure on, compact tables) because pypdf and PyMuPDF flatten tables and turn figures into noise, which no cleaning step can undo. Docling keeps table structure and isolates figures; its remaining gaps (figure content, merged cells) are fixed with a vision caption and by keeping each table's caption in its chunk. Cleaning (NFKC, hyphenation, header/footer removal) still runs on its output.
