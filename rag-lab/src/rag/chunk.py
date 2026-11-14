"""rag/chunk.py: parsed documents -> chunks with metadata (Phase 02, Steps 3-4).

Input:  data/parsed/<parser>/<doc_id>.json      (from rag/parse.py)
Output: data/chunks/<parser>-<chunker>.jsonl   (one chunk per line)

Chunkers (one interface: chunk(doc) -> list[Chunk]):
    recursive   3.1  split the whole document by \\n\\n, \\n, ". ", " " until each piece fits
    structure   3.2  one section at a time, section path prepended, tables split by rows
                     with the header repeated, figures as their own chunks
    (semantic 3.3 and contextual 3.6 come next: they need LM Studio)

Sizes are in TOKENS of the embedding model's tokenizer, never characters.

Usage (from rag-lab/):
    uv run python -m rag.chunk recursive --size 512 --overlap 64
    uv run python -m rag.chunk structure --size 512 --show 3
    uv run python -m rag.chunk recursive --parser pypdf      # baseline input
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

from rag.paths import CHUNKS as CHUNKS_DIR
from rag.paths import parsed_dir
DOC_VERSION = 1  # bumped per document when we handle updates (Phase 08)

# Embedding model (from .env) -> its tokenizer on Hugging Face (tokenizer files only)
TOKENIZERS = {
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
}


# ---------- tokens ----------

@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    model = os.getenv("EMBED_MODEL", "nomic").lower()
    name = next((hf for key, hf in TOKENIZERS.items() if key in model), TOKENIZERS["nomic"])
    tok = AutoTokenizer.from_pretrained(name)
    tok.model_max_length = 10**9  # we count long texts on purpose; silence the warning
    return tok


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


# ---------- chunk schema (Step 4) ----------

@dataclass
class Chunk:
    chunk_id: str            # deterministic: <doc_id>#v<version>#p<page>#c<n>
    doc_id: str
    doc_version: int
    parent_id: str           # top-level section, for parent-child retrieval (Phase 05)
    text: str                # exactly what gets embedded and sent to the LLM
    element_type: str        # text | table | figure
    section_path: str        # "Title > 2 Method > 2.1 Setup"
    page: int                # first page, for citations (golden.jsonl matches on it)
    pages: list[int]         # every page the chunk touches
    source: str              # file name, matches golden.jsonl "doc"
    source_uri: str
    title: str
    chunker: str
    n_tokens: int
    content_hash: str        # of text: detect changed chunks on re-ingestion
    context: str | None = None      # contextual retrieval sentence (3.6)
    image_path: str | None = None   # figures: for vision captioning (Step 5)


class Chunker(Protocol):
    name: str

    def chunk(self, doc: dict) -> list[Chunk]: ...


class ChunkBuilder:
    """Collects one document's chunks and gives them deterministic ids."""

    def __init__(self, doc: dict, chunker: str) -> None:
        self.doc = doc
        self.chunker = chunker
        self.chunks: list[Chunk] = []
        self._parents: dict[str, str] = {}

    def parent_id(self, path: list[str]) -> str:
        top = " > ".join(path[:2])  # title + top-level section
        if top not in self._parents:
            self._parents[top] = f"{self.doc['doc_id']}#v{DOC_VERSION}#s{len(self._parents):02d}"
        return self._parents[top]

    def add(self, text: str, pages: set[int], path: list[str],
            element_type: str = "text", image_path: str | None = None) -> None:
        text = text.strip()
        if not text:
            return
        page = min(pages) if pages else 0
        doc = self.doc
        self.chunks.append(Chunk(
            chunk_id=f"{doc['doc_id']}#v{DOC_VERSION}#p{page}#c{len(self.chunks):03d}",
            doc_id=doc["doc_id"],
            doc_version=DOC_VERSION,
            parent_id=self.parent_id(path),
            text=text,
            element_type=element_type,
            section_path=" > ".join(path),
            page=page,
            pages=sorted(pages),
            source=Path(doc["source_uri"]).name,
            source_uri=doc["source_uri"],
            title=doc["title"],
            chunker=self.chunker,
            n_tokens=count_tokens(text),
            content_hash="sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16],
            image_path=image_path,
        ))


# ---------- recursive splitting on character spans ----------
# Working with (start, end) spans instead of strings lets us map every chunk
# back to the elements (and so the pages) it came from.

SEPARATORS = ["\n\n", "\n", ". ", " "]
Span = tuple[int, int]


def split_spans(text: str, start: int, end: int, max_tokens: int,
                separators: list[str] = SEPARATORS) -> list[Span]:
    """Cut text[start:end] into spans of <= max_tokens, biggest separator first."""
    if count_tokens(text[start:end]) <= max_tokens:
        return [(start, end)]
    for i, sep in enumerate(separators):
        cuts = [start + m.end() for m in re.finditer(re.escape(sep), text[start:end])]
        cuts = [c for c in cuts if c < end]
        if cuts:
            bounds = [start, *cuts, end]
            spans: list[Span] = []
            for a, b in zip(bounds, bounds[1:]):
                spans += split_spans(text, a, b, max_tokens, separators[i + 1:])
            return spans
    return hard_split(text, start, end, max_tokens)


def hard_split(text: str, start: int, end: int, max_tokens: int) -> list[Span]:
    """Last resort (no separator left, e.g. a giant URL): cut every max_tokens tokens."""
    offsets = tokenizer()(text[start:end], add_special_tokens=False,
                          return_offsets_mapping=True)["offset_mapping"]
    cuts = [start + offsets[i][0] for i in range(max_tokens, len(offsets), max_tokens)]
    bounds = [start, *cuts, end]
    return list(zip(bounds, bounds[1:]))


def merge_spans(text: str, spans: list[Span], max_tokens: int, overlap: int) -> list[Span]:
    """Greedily pack small spans into chunks of <= max_tokens; repeat ~overlap tokens."""
    sizes = [count_tokens(text[a:b]) for a, b in spans]
    merged: list[Span] = []
    i = 0
    while i < len(spans):
        j, total = i, 0
        while j < len(spans) and total + sizes[j] <= max_tokens:
            total += sizes[j]
            j += 1
        j = max(j, i + 1)
        merged.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        # next chunk starts a few spans back, so it repeats up to `overlap` tokens
        k, back = j, 0
        while k - 1 > i and back + sizes[k - 1] <= overlap:
            k -= 1
            back += sizes[k]
        i = k
    return merged


def join_elements(elements: list[dict]) -> tuple[str, list[tuple[int, int, dict]]]:
    """Concatenate element texts with blank lines; remember each element's span."""
    parts, offsets, pos = [], [], 0
    for e in elements:
        if not e["text"]:
            continue
        offsets.append((pos, pos + len(e["text"]), e))
        parts.append(e["text"])
        pos += len(e["text"]) + 2
    return "\n\n".join(parts), offsets


def chunk_text(builder: ChunkBuilder, elements: list[dict], max_tokens: int, overlap: int,
               header: str = "", path: list[str] | None = None) -> None:
    """Recursive split + merge of a run of elements, adding the chunks to the builder."""
    text, offsets = join_elements(elements)
    if not text:
        return
    budget = max_tokens - (count_tokens(header) + 1 if header else 0)
    for a, b in merge_spans(text, split_spans(text, 0, len(text), budget), budget, overlap):
        touched = [e for s, t, e in offsets if s < b and t > a]
        chunk_path = path if path is not None else (touched[0]["section_path"] if touched else [])
        body = text[a:b].strip()
        builder.add(f"{header}\n{body}" if header else body,
                    pages={e["page"] for e in touched}, path=chunk_path)


# ---------- 3.1 recursive ----------

class RecursiveChunker:
    """Whole document as one stream; size-driven. The sensible default baseline."""

    def __init__(self, max_tokens: int = 512, overlap: int = 64) -> None:
        self.max_tokens, self.overlap = max_tokens, overlap
        self.name = f"recursive-{max_tokens}-{overlap}"

    def chunk(self, doc: dict) -> list[Chunk]:
        builder = ChunkBuilder(doc, self.name)
        chunk_text(builder, doc["elements"], self.max_tokens, self.overlap)
        return builder.chunks


# ---------- 3.2 structure-aware ----------

def group_sections(elements: list[dict]) -> list[tuple[list[str], list[dict]]]:
    """Consecutive elements that share a section path."""
    groups: list[tuple[list[str], list[dict]]] = []
    for e in elements:
        if groups and groups[-1][0] == e["section_path"]:
            groups[-1][1].append(e)
        else:
            groups.append((e["section_path"], [e]))
    return groups


def split_table(table: dict, budget: int) -> list[str]:
    """Whole table if it fits; else row groups that each repeat caption + header row."""
    if count_tokens(table["text"]) <= budget:
        return [table["text"]]
    lines = table["markdown"].splitlines()
    head, rows = lines[:2], lines[2:]
    prefix = "\n".join(([table["caption"], ""] if table.get("caption") else []) + head)
    room = max(budget - count_tokens(prefix), 1)

    pieces, current, used = [], [], 0
    for row in rows:
        size = count_tokens(row) + 1
        if current and used + size > room:
            pieces.append(current)
            current, used = [], 0
        current.append(row)
        used += size
    pieces.append(current)
    return [prefix + "\n" + "\n".join(p) for p in pieces]


class StructureChunker:
    """One section at a time; never mixes sections; tables and figures are their own chunks."""

    def __init__(self, max_tokens: int = 512, overlap: int = 64) -> None:
        self.max_tokens, self.overlap = max_tokens, overlap
        self.name = f"structure-{max_tokens}-{overlap}"

    def chunk(self, doc: dict) -> list[Chunk]:
        builder = ChunkBuilder(doc, self.name)
        for path, elements in group_sections(doc["elements"]):
            header = " > ".join(path)  # "Title > 2 Method > 2.1 Setup": cheap, very effective
            budget = self.max_tokens - count_tokens(header) - 1
            buffer: list[dict] = []

            def flush() -> None:
                chunk_text(builder, buffer, self.max_tokens, self.overlap, header, path)
                buffer.clear()

            for e in elements:
                if e["type"] == "heading":
                    continue  # already in the header line
                if e["type"] == "table":
                    flush()
                    for piece in split_table(e, budget):
                        builder.add(f"{header}\n{piece}", {e["page"]}, path, "table")
                elif e["type"] == "figure":
                    flush()
                    if e["text"]:  # caption only for now; vision caption in Step 5
                        builder.add(f"{header}\n{e['text']}", {e["page"]}, path, "figure",
                                    image_path=e.get("image_path"))
                else:
                    buffer.append(e)
            flush()
        return builder.chunks


# ---------- 3.3 semantic ----------

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "etc.", "Fig.", "Eq.", "vs.", "cf.", "Sec.", "No.")


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that doesn't break after "e.g." or "et al."."""
    sentences: list[str] = []
    for part in SENTENCE_END.split(text):
        if sentences and sentences[-1].endswith(ABBREVIATIONS):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class Sentence:
    text: str
    page: int
    path: list[str]
    n_tokens: int


def find_breakpoints(sentences: list[Sentence], percentile: float, window: int) -> set[int]:
    """Indices i where a new chunk starts after sentence i (topic shift).

    Each sentence is embedded together with its neighbours (window) to smooth
    noise; a break goes where the cosine distance between neighbouring windows
    is above the document's `percentile`-th distance.
    """
    import numpy as np

    from rag.embed import embed_docs

    if len(sentences) < 3:
        return set()
    windows = [" ".join(s.text for s in sentences[max(0, i - window): i + window + 1])
               for i in range(len(sentences))]
    vectors = embed_docs(windows)                                # normalized: dot = cosine
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)   # distance(i, i+1)
    threshold = np.percentile(distances, percentile)
    return {i for i, d in enumerate(distances) if d > threshold}


def pack_sentences(sentences: list[Sentence], max_tokens: int) -> list[list[Sentence]]:
    """Split a too-long semantic group at sentence boundaries."""
    groups, current, used = [], [], 0
    for s in sentences:
        if current and used + s.n_tokens > max_tokens:
            groups.append(current)
            current, used = [], 0
        current.append(s)
        used += s.n_tokens + 1
    if current:
        groups.append(current)
    return groups


class SemanticChunker:
    """Breaks where the meaning shifts, measured with embeddings. No size target, only a cap.

    Tables and figures are kept whole (splitting a Markdown table into
    "sentences" makes no sense), exactly as in the structure-aware chunker.
    """

    def __init__(self, max_tokens: int = 512, overlap: int = 0,
                 percentile: float = 95, window: int = 1, min_tokens: int = 50) -> None:
        self.max_tokens, self.percentile = max_tokens, percentile
        self.window, self.min_tokens = window, min_tokens
        self.name = f"semantic-p{percentile:g}-{max_tokens}"

    def chunk(self, doc: dict) -> list[Chunk]:
        builder = ChunkBuilder(doc, self.name)
        sentences: list[Sentence] = []
        forced: set[int] = set()        # break after these: a table/figure sits between them
        atomic: dict[int, list[dict]] = {}  # tables/figures to emit after sentence index i

        for e in doc["elements"]:
            if e["type"] in ("table", "figure"):
                forced.add(len(sentences) - 1)
                atomic.setdefault(len(sentences) - 1, []).append(e)
                continue
            for text in split_sentences(e["text"]):
                sentences.append(Sentence(text, e["page"], e["section_path"], count_tokens(text)))

        breaks = find_breakpoints(sentences, self.percentile, self.window) | forced
        for e in atomic.get(-1, []):    # tables/figures before the first sentence
            self._add_atomic(builder, e)

        group: list[Sentence] = []
        for i, s in enumerate(sentences):
            group.append(s)
            if i in breaks or i == len(sentences) - 1:
                # tiny topic islands make poor retrieval targets: keep growing unless forced
                if sum(x.n_tokens for x in group) >= self.min_tokens or i in forced \
                        or i == len(sentences) - 1:
                    self._add_group(builder, group)
                    group = []
                for e in atomic.get(i, []):
                    self._add_atomic(builder, e)
        return builder.chunks

    def _add_group(self, builder: ChunkBuilder, group: list[Sentence]) -> None:
        for part in pack_sentences(group, self.max_tokens):
            builder.add(" ".join(s.text for s in part), {s.page for s in part}, part[0].path)

    def _add_atomic(self, builder: ChunkBuilder, e: dict) -> None:
        if e["type"] == "table":
            for piece in split_table(e, self.max_tokens):
                builder.add(piece, {e["page"]}, e["section_path"], "table")
        elif e["text"]:
            builder.add(e["text"], {e["page"]}, e["section_path"], "figure",
                        image_path=e.get("image_path"))


CHUNKERS = {"recursive": RecursiveChunker, "structure": StructureChunker, "semantic": SemanticChunker}


# ---------- run over the corpus ----------

def load_docs(parser: str) -> list[dict]:
    """Every document in data/parsed/<parser>/."""
    folder = parsed_dir(parser)
    files = sorted(folder.glob("*.json"))
    if not files:
        raise SystemExit(f"No parsed docs in {folder}. Run: uv run python -m rag.parse data/raw --parser {parser}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def write_chunks(chunks: list[Chunk], name: str) -> Path:
    out = CHUNKS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in chunks) + "\n",
                   encoding="utf-8")
    return out


def print_stats(chunks: list[Chunk], max_tokens: int) -> None:
    tokens = [c.n_tokens for c in chunks]
    print(f"chunks      {len(chunks)}")
    print(f"tokens      avg {statistics.mean(tokens):.0f}, median {statistics.median(tokens):.0f}, "
          f"min {min(tokens)}, max {max(tokens)}")
    print(f"over limit  {sum(t > max_tokens for t in tokens)} (> {max_tokens} tokens)")
    print(f"tiny        {sum(t < 50 for t in tokens)} (< 50 tokens)")
    print(f"types       {dict(Counter(c.element_type for c in chunks))}")
    print(f"per doc     {dict(Counter(c.source for c in chunks))}")


def build_chunker(kind: str, args: argparse.Namespace) -> Chunker:
    if kind == "semantic":
        return SemanticChunker(args.size, percentile=args.percentile)
    return CHUNKERS[kind](args.size, args.overlap)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk parsed documents.")
    ap.add_argument("chunker", choices=[*CHUNKERS, "contextual"])
    ap.add_argument("--size", type=int, default=512, help="max tokens per chunk")
    ap.add_argument("--overlap", type=int, default=64, help="overlap in tokens")
    ap.add_argument("--parser", default="docling", help="which parsed output to read")
    ap.add_argument("--show", type=int, default=0, help="print N random chunks to eyeball")
    semantic = ap.add_argument_group("semantic")
    semantic.add_argument("--percentile", type=float, default=95, help="break above this distance percentile")
    contextual = ap.add_argument_group("contextual")
    contextual.add_argument("--base", choices=CHUNKERS, default="structure", help="chunker to enrich")
    contextual.add_argument("--model", default=None, help="LLM for contexts (default: HELPER_MODEL)")
    contextual.add_argument("--workers", type=int, default=1,
                            help="parallel LLM requests (share one context budget: keep 1 for long docs)")
    contextual.add_argument("--limit", type=int, default=None, help="enrich only N chunks per doc (timing)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    chunker = build_chunker(args.chunker if args.chunker != "contextual" else args.base, args)
    if args.chunker == "contextual":
        from rag.enrich import ContextualChunker

        chunker = ContextualChunker(chunker, args.model, args.workers, args.limit)

    chunks = [c for doc in load_docs(args.parser) for c in chunker.chunk(doc)]

    name = f"{args.parser}-{chunker.name}" + (f"-limit{args.limit}" if args.limit else "")
    out = write_chunks(chunks, name)
    print(f"\n{name} -> {out}\n")
    print_stats(chunks, args.size)
    if stats := getattr(chunker, "stats", None):
        rate = stats["llm_calls"] / stats["seconds"] * 60 if stats["seconds"] and stats["llm_calls"] else 0
        print(f"llm         {stats['llm_calls']} calls, {stats['cache_hits']} cached, "
              f"{stats['seconds']:.0f}s ({rate:.0f} chunks/min)")

    for c in random.Random(0).sample(chunks, min(args.show, len(chunks))):
        print(f"\n----- {c.chunk_id}  [{c.element_type}, {c.n_tokens} tok, pages {c.pages}]\n{c.text}")


if __name__ == "__main__":
    main()
