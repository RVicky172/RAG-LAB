# Phase 02: Data Ingestion & Chunking Engineering

**Goal:** Master the "garbage in, garbage out" bottleneck. Retrieval quality is capped by how well documents are parsed and split.
**Time:** about 2 weeks part-time.
**Guide sections:** Part 1 §3 (Parsing), §4 (Chunking); Part 2 §7 (Production ingestion).

**Models this phase** (see [MODELS.md](../MODELS.md)):

| Role | Model | Why here |
|------|-------|----------|
| Contextual retrieval, propositions | **Qwen3-4B-Instruct-2507** (LM Studio) | One call per chunk means thousands of calls; a small model fully in VRAM makes that practical |
| Figure and chart captions | **Qwen3.6-35B-A3B** + mmproj (LM Studio, vision) | Reads images; the meaning of charts is in the picture, not the text layer |
| Embeddings (compared; **nomic kept**, see Outcome) | **Qwen3-Embedding-0.6B** (LM Studio or sentence-transformers) vs nomic-embed-v1.5 | 32K-token input means no silent truncation of large chunks; multilingual; compare it with nomic in the ablation |
| Semantic chunking | the same embedding model | Sentence-to-sentence similarity finds topic boundaries |

---

## Why this phase matters

Look at your `failures.md` from Phase 01. Most failures are probably here: tables flattened into number soup, two-column pages read straight across, headers and footers in every chunk, a sentence cut in half, or a chunk that says "it rose 20%" without saying *what* rose. **No embedding model or reranker can recover meaning that parsing destroyed.**

Real AI engineers spend more time here than on any model choice.

---

## Learning steps (in order)

### Step 1: Look at your data (seriously)

Before writing any parser:
1. Open 10 representative documents. Classify them: digital PDF prose, scanned PDF, multi-column, table-heavy, figure-heavy, HTML, Markdown, DOCX.
2. Print what `pypdf` extracted for 5 tricky pages. Compare with the visual page.
3. Write down every defect you see. This list drives your parser choice.

### Step 2: Document parsing & extraction

**Choose the parser by document type:**

| Document type | Approach | Tools |
|---------------|----------|-------|
| Digital PDF, mostly prose | Text-layer extraction | PyMuPDF (`fitz`), pdfplumber |
| Scanned PDF / photo | OCR | Tesseract, Textract, a vision LLM |
| Multi-column, tables, forms | Layout-aware parsing (reading order + table structure) | **Docling**, Unstructured (`hi_res`), Textract LAYOUT+TABLES, Marker |
| Charts, diagrams | Region detection + vision-LLM captions, or multimodal embeddings | Qwen3.6-35B-A3B (local vision), ColQwen/ColPali (Phase 07) |
| HTML | Structure-aware: keep headings, drop nav/boilerplate | BeautifulSoup, trafilatura |
| Markdown / DOCX | Keep heading hierarchy | markdown-it, python-docx, Docling |

**Target output: typed elements, not a text blob.**

```json
{"doc_id": "hr-leave-policy-2026", "elements": [
  {"type": "heading",   "level": 2, "text": "4. Sick Leave", "page": 3},
  {"type": "paragraph", "text": "Employees accrue 1 day per month...", "page": 3,
   "section_path": ["HR Leave Policy", "4. Sick Leave"]},
  {"type": "table", "markdown": "| Grade | Days |\n|---|---|\n| L1-L3 | 12 |", "page": 4},
  {"type": "figure", "image_path": "data/processed/figs/p5-fig1.png", "page": 5}
]}
```

**Cleaning checklist:**
- Remove repeated headers and footers (detect lines that recur on most pages)
- Fix hyphenation across line breaks (`reim-\nbursement` → `reimbursement`)
- Normalize whitespace and Unicode (ligatures like `ﬁ`)
- Keep page numbers for citations
- Represent tables as Markdown or HTML, never as flattened text
- Deduplicate near-identical documents (hash, or MinHash for near-duplicates)

#### Lab 2a: Parser shoot-out

Parse the same 5 tricky pages with **PyMuPDF**, **Docling** and **Unstructured**. Save the output side by side and score each 1–5 on reading order, tables and headings. Build `rag/parse.py` with one interface:

```python
class Parser(Protocol):
    def parse(self, path: str) -> dict: ...   # returns {"doc_id", "elements": [...]}
```

> **Rule:** always eyeball parser output for 5–10 representative pages before indexing a whole corpus.

### Step 3: Chunking strategies

A good chunk is **self-contained** (makes sense alone), **focused** (one idea, so its embedding is sharp) and **traceable** (carries metadata to its source).

#### 3.1 Fixed-size and recursive character splitting
- **Fixed:** every N tokens with overlap. A baseline; cuts sentences and tables mid-way.
- **Recursive:** try to split on `\n\n` (paragraphs), then `\n`, then `. `, then spaces, until each piece fits. The sensible default.
- **Size in tokens, not characters.** Balance:
  - Too small → loses context ("it", "this policy")
  - Too large → embedding becomes a blurry average; wastes LLM context
  - Must be ≤ the embedding model's max input (often 512 tokens for small models; 8K for others)
- **Starting point:** 300–800 tokens, 10–20% overlap. Then *tune on the golden set*.

#### 3.2 Document-structure-aware
- Split by Markdown headers, HTML tags or parser headings: one chunk per section, recursively split if too big.
- Code: split by AST (functions, classes) using `tree-sitter`.
- **Prepend the section path to each chunk:** `"HR Leave Policy > 4. Sick Leave\n<text>"`. This is cheap and very effective.
- Tables: keep whole if they fit; otherwise split by rows and **repeat the header row** in every piece.

#### 3.3 Semantic chunking
- Embed each sentence; compute similarity between neighbours; start a new chunk where similarity drops below a threshold (e.g. a 95th-percentile distance jump).
- Good for unstructured text (transcripts, emails, chats). Costs extra embedding calls at ingestion. Often *not* better than structure-aware on well-structured documents, so measure it.

#### 3.4 Propositional chunking
- An LLM rewrites text into atomic, standalone facts ("Employees at grade L2 receive 12 sick days per year.").
- Very precise retrieval for fact lookups; expensive; loses narrative flow. Usually indexed *alongside* normal chunks, pointing back to the parent.

#### 3.5 Late chunking
- Run a long-context embedding model over the **whole document** (token-level embeddings), *then* pool token embeddings per chunk span. Each chunk vector carries document-wide context.
- Needs a long-context embedding model that exposes token embeddings (e.g. jina-embeddings-v3). Good to know conceptually and for interviews.

#### 3.6 Contextual Retrieval (enrichment; high value)
- For each chunk, an LLM writes 1–2 sentences situating it in the document ("This chunk is from ACME's Q3 2026 report, section on operating margin."). Prepend it before **both** embedding and BM25 indexing.
- Anthropic's published benchmark: about **49% fewer retrieval failures** (contextual embeddings + contextual BM25), about **67%** with a reranker added. Treat that as evidence the technique works, not a guaranteed number for your data.
- Cost: one LLM call per chunk. Locally, the cost is **time**: use the 4B helper model, keep the document as an **identical prefix** for every chunk of that document (LM Studio / llama.cpp can then reuse the cached prompt prefix instead of re-reading it), and for long documents pass the section plus a short document summary instead of the whole document. In the cloud (Phase 09), the same prefix design enables provider prompt caching.

```python
# rag/enrich.py
from rag.llm import helper

def contextualize(doc_text: str, chunk_text: str) -> str:
    prompt = (f"<document>\n{doc_text}\n</document>\n"          # same prefix for every chunk of this doc
              f"<chunk>\n{chunk_text}\n</chunk>\n"
              "Write 1-2 sentences that situate this chunk within the document "
              "to improve search retrieval. Reply with the sentences only.")
    return helper(prompt, max_tokens=150)
```

Time it: chunks per minute with Qwen3-4B vs Qwen3-30B-A3B. Then estimate how long your whole corpus takes.

#### 3.7 Parent-child / hierarchical (preview)
Index small child chunks for precise matching; return their larger parent section to the LLM. You'll build this in Phase 05. Design your chunk IDs for it now (`parent_id`).

| Strategy | Use when | Trade-off |
|----------|----------|-----------|
| Fixed-size | Uniform text, logs, a baseline | Cuts sentences and tables |
| Recursive | Default for most documents | Still size-driven at the leaves |
| Structure-aware | Policies, manuals, contracts, code | Uneven section sizes |
| Semantic | Transcripts, emails, chats | Extra embedding cost |
| Propositional | Dense fact lookup | LLM cost; loses flow |
| Late chunking | Long docs where chunks depend on global context | Needs a special model |
| Contextual retrieval | Chunks ambiguous alone | One LLM call per chunk |
| No chunking | FAQs, product cards | Long files exceed limits |

### Step 4: Metadata enrichment

Metadata powers citations, filters, updates and security. Use this schema on every chunk:

```json
{
  "chunk_id": "hr-leave-policy-2026#v3#p3#c07",
  "doc_id": "hr-leave-policy-2026", "doc_version": 3,
  "parent_id": "hr-leave-policy-2026#v3#s4",
  "text": "Employees accrue 1 day of sick leave per month...",
  "context": "From the 2026 HR Leave Policy, section 4 (Sick Leave).",
  "element_type": "paragraph",
  "section_path": "HR Leave Policy > 4. Sick Leave",
  "page": 3, "source_uri": "data/raw/hr/leave-policy-2026.pdf",
  "title": "HR Leave Policy 2026", "author": "HR", "doc_date": "2026-01-15",
  "tenant_id": "acme", "allowed_groups": ["all-employees"],
  "language": "en", "content_hash": "sha256:...",
  "keywords": ["sick leave", "accrual"], "summary": "...",
  "embedding_model": "BAAI/bge-small-en-v1.5", "chunker": "structure-v2"
}
```

| Field | Why it exists |
|-------|---------------|
| `page`, `source_uri` | Citations |
| `tenant_id`, `allowed_groups` | Security filters (Phase 08) |
| `doc_id`, `doc_version`, `content_hash` | Idempotent updates and deletions |
| `doc_date`, `element_type`, `section_path` | Filtering and self-query (Phase 04) |
| `embedding_model`, `chunker` | Lineage: know what to rebuild |

LLM-generated `summary` and `keywords` (or "questions this chunk answers") can be embedded as extra retrieval keys pointing to the same chunk.

### Step 5: Images, tables and multimodal content (intro)

- **Tables:** Markdown/HTML; keep whole or row-split with headers; optionally embed an LLM summary but send the raw table to the generator.
- **Figures:** send the image + section title + nearby text to a vision LLM for a 3–5 sentence caption; store it as a chunk with `element_type: figure` and a link to the image. Locally, load **Qwen3.6-35B-A3B** (it has the `mmproj` vision file) only while captioning:

```python
# rag/caption.py: image captioning via LM Studio's OpenAI-compatible API
import base64, os
from openai import OpenAI
client = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")

def caption_figure(png_path: str, section_title: str, nearby_text: str) -> str:
    b64 = base64.b64encode(open(png_path, "rb").read()).decode()
    r = client.chat.completions.create(
        model=os.getenv("VISION_MODEL"), temperature=0, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": f"Section: {section_title}\nNearby text: {nearby_text[:1500]}\n"
                                     "Describe this figure for search: what it shows, axes, key values "
                                     "and the main takeaway. 3-5 sentences."}]}])
    return r.choices[0].message.content.strip()
```
- Phase 07 goes deeper (ColPali, multimodal embeddings).

### Step 6: Measure (the ablation)

Run every chunker over your corpus and evaluate with the golden set:

```
| Config                                      | #chunks | avg tok | hit@5 | MRR  |
|---------------------------------------------|---------|---------|-------|------|
| 01 naive 800-char                           |  2,140  |   190   | 0.60  | 0.42 |
| 02a recursive 512 tok / 64 overlap          |         |         |       |      |
| 02b structure-aware + section path          |         |         |       |      |
| 02c semantic                                |         |         |       |      |
| 02d structure-aware + contextual retrieval  |         |         |       |      |
```

Also try 2–3 chunk sizes (256 / 512 / 1,024) for your best strategy, and the **embedding model** as its own variable: nomic-embed-v1.5 vs Qwen3-Embedding-0.6B (and 4B if time allows). Change one thing at a time.

---

## Milestone

- `rag/parse.py`: layout-aware parser producing typed elements with page numbers; header/footer removal
- `rag/chunk.py`: at least 4 chunkers (recursive, structure-aware with table row-splitting, semantic, contextual enrichment) behind one interface
- Every chunk carries the metadata schema above, with **deterministic chunk IDs**
- Ablation table in `results.md` and a one-paragraph decision: which chunker you chose and why
- Grow `golden.jsonl` to **40 questions**, with more tables and figures

## Outcome (completed 2026-11-14)

**Chosen pipeline:** Docling (OCR off) → structure-aware chunks, **400 / 64 tokens**, section path prepended, tables row-split with the header repeated → figure chunks with a **vision-LLM description** (Qwen3.6-35B-A3B) → **nomic-embed-text-v1.5**.
Golden v4 (53 answerable): **hit@5 0.92, MRR 0.80**, chart questions fig@5 **0.93** (Phase 02 start: pypdf recursive 0.85 / 0.71).

| What was tried | Verdict |
|----------------|---------|
| Docling vs pypdf vs PyMuPDF | Docling: reading order, tables, figure crops |
| Recursive / structure-aware / semantic | Structure-aware; semantic dropped (no gain, extra cost) |
| Contextual retrieval (Qwen3-4B) | Not adopted: +0.05 MRR at 512, −0.06 at 400. Retry with BM25 (Phase 03) |
| Vision captions | **Adopted:** chart fig@5 0.53 → 0.93 |
| Qwen3-Embedding-0.6B / 4B vs nomic | nomic kept. 0.6B 0.85 / 0.74; 4B ties (0.91 / 0.80) at 30x the size, but solves different questions → second retriever in Phase 03 |
| Chunk size 256 / 400 / 512 / 1024 | 400 (256–512 flat, 1024 −7 points) |

Not done: propositional and late chunking (read only), table summaries (tables already score 1.00). Open misses: q013, q021 (paraphrase), q023 (topic-word trap), q048 (description dropped the figure's example text). Full evidence: [rag-lab/eval/phase02_chunking/results.md](../../rag-lab/eval/phase02_chunking/results.md), findings 1–30.

## Exit checklist

- [x] I can pick a parser for a given document type and explain why. (Lab 2a shoot-out; two-column papers)
- [x] I can explain why chunk size is a trade-off and how I'd tune it. (findings 29–30)
- [x] I know when semantic chunking helps and when it's wasted cost. (semantic: no gain on structured papers)
- [x] I can explain contextual retrieval and late chunking. (contextual: measured, size-dependent; late chunking: read only)
- [x] My tables never get split without their header row. (`split_table`, `test_table_split_repeats_caption_and_header`)
- [x] Every chunk has page, section, doc version and hash metadata. (`Chunk` schema, deterministic ids, `test_chunk_ids_are_deterministic`)

## Common pitfalls

- Choosing a chunk size "because a blog said 512" without measuring.
- Losing the table header when splitting a table.
- Measuring chunk size in characters against a token limit.
- Indexing a corpus without reading the parser output first.
- Random chunk IDs (uuid4): re-runs create duplicates instead of overwriting.

## Interview questions

1. You get a 400-page PDF. How do you chunk it?
2. How do you choose chunk size and overlap?
3. How do you handle tables and images?
4. A chunk says "revenue grew 12%" but not whose revenue. How do you fix it?
5. What's late chunking, and how is it different from contextual retrieval?

## Resources

- Docling (github.com/docling-project/docling), Unstructured docs, PyMuPDF docs
- Anthropic: *Introducing Contextual Retrieval*
- Greg Kamradt: *5 Levels of Text Splitting* (notebook/video)
- Chen et al., 2023: *Dense X Retrieval: What Retrieval Granularity Should We Use?* (propositions)
- Jina AI, 2024: *Late Chunking* paper
- Guide: Part 1 §3, §4; Interview Q4–Q8

## Next

→ [Phase 03: Vector Databases & Indexing Mechanics](../03-vector-search-indexing/README.md)
