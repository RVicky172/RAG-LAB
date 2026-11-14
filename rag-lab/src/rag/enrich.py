"""rag/enrich.py: Contextual Retrieval (Phase 02, Step 3.6).

For every chunk, the helper LLM writes 1-2 sentences that situate the chunk in
its document ("This chunk is from 'Lost in the Middle', section 2.3, and reports
closed-book vs oracle accuracy..."). The sentences are prepended to the chunk
text, so they are part of both the embedding and (Phase 03) BM25 index.

Speed tricks:
- The document is the SAME prefix for every chunk of that document, so LM Studio
  (llama.cpp) can reuse the cached prompt instead of re-reading 10K tokens each time.
- The reference list is left out of the document: long, and useless as context.
- Results are cached on disk by (model, document hash, chunk hash): re-runs and
  other chunkers that produce the same chunk text cost nothing.

Usage (from rag-lab/):
    uv run python -m rag.chunk contextual --size 512            # wraps the structure chunker
    uv run python -m rag.chunk contextual --model qwen/qwen3-30b-a3b-2507 --limit 10  # timing
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from rag.chunk import Chunk, Chunker, count_tokens, tokenizer
from rag.llm import HELPER_MODEL, complete

from rag.paths import CONTEXTS_CACHE as CACHE_FILE
MAX_DOC_TOKENS = 20_000  # helper model context is ~29K; leave room for chunk + answer
SKIP_SECTIONS = re.compile(r"^(references|bibliography)$", re.IGNORECASE)

# Document first, chunk last: everything before <chunk> is identical for every
# chunk of the same document, which is what makes prefix caching work.
PROMPT = """<document>
{document}
</document>
Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>
Write 1-2 short sentences that situate this chunk within the overall document, to improve \
search retrieval of the chunk. Name the document and say what the chunk is about \
(e.g. which method, experiment, table or result). Reply with the sentences only."""


def document_text(doc: dict) -> str:
    """The document as context: no reference list, capped at MAX_DOC_TOKENS."""
    parts = [e["text"] for e in doc["elements"]
             if e["text"] and not (len(e["section_path"]) > 1 and SKIP_SECTIONS.match(e["section_path"][1]))]
    text = f"Title: {doc['title']}\n\n" + "\n\n".join(parts)
    offsets = tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) > MAX_DOC_TOKENS:
        text = text[: offsets[MAX_DOC_TOKENS][0]] + "\n[... document truncated ...]"
    return text


class ContextCache:
    """Append-only JSONL cache: {"key": ..., "context": ...} per line. Thread-safe."""

    def __init__(self, path: Path = CACHE_FILE) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.data[row["key"]] = row["context"]

    @staticmethod
    def key(model: str, doc: dict, chunk: Chunk) -> str:
        raw = f"{model}|{doc['content_hash']}|{chunk.content_hash}|{PROMPT}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, context: str) -> None:
        with self.lock:
            self.data[key] = context
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "context": context}, ensure_ascii=False) + "\n")


class ContextualChunker:
    """Wraps any chunker and prepends an LLM-written context to each of its chunks."""

    def __init__(self, base: Chunker, model: str | None = None, workers: int = 1,
                 limit: int | None = None) -> None:
        self.base = base
        self.model = model or HELPER_MODEL
        # Parallel requests share ONE context budget (KV cache) in LM Studio: 4 x 10K-token
        # documents overflow a 29K context (HTTP 400). Sequential also keeps the cached
        # document prefix warm. Raise only for short documents or a bigger loaded context.
        self.workers = workers
        self.limit = limit                # only enrich the first N chunks per doc (timing runs)
        self.cache = ContextCache()
        short_model = self.model.split("/")[-1]
        self.name = f"contextual-{short_model}-{base.name}"
        self.stats = {"llm_calls": 0, "cache_hits": 0, "seconds": 0.0}

    def contextualize(self, document: str, doc: dict, chunk: Chunk) -> str:
        key = self.cache.key(self.model, doc, chunk)
        if (cached := self.cache.get(key)) is not None:
            self.stats["cache_hits"] += 1
            return cached
        context = complete(PROMPT.format(document=document, chunk=chunk.text),
                           model=self.model, max_tokens=150, temperature=0)
        context = " ".join(context.split())  # one line, no stray newlines
        self.cache.put(key, context)
        self.stats["llm_calls"] += 1
        return context

    def chunk(self, doc: dict) -> list[Chunk]:
        chunks = self.base.chunk(doc)
        if self.limit:
            chunks = chunks[: self.limit]
        document = document_text(doc)

        start = time.perf_counter()
        with ThreadPoolExecutor(self.workers) as pool:
            contexts = list(pool.map(lambda c: self.contextualize(document, doc, c), chunks))
        seconds = time.perf_counter() - start
        self.stats["seconds"] += seconds
        print(f"  {doc['doc_id']}: {len(chunks)} chunks in {seconds:.0f}s "
              f"({len(chunks) / max(seconds, 1e-9) * 60:.0f} chunks/min)", flush=True)

        enriched = []
        for c, context in zip(chunks, contexts):
            text = f"{context}\n\n{c.text}"
            enriched.append(replace(
                c, text=text, context=context, chunker=self.name, n_tokens=count_tokens(text),
                content_hash="sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16],
            ))
        return enriched
