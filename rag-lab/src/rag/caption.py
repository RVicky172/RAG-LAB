"""rag/caption.py: vision-LLM figure descriptions (Phase 02, Step 5).

Why: a figure chunk used to hold only its printed caption ("Figure 5: The
effect of changing the position of relevant information..."). The caption
names the figure but rarely states what it shows: the block labels in an
architecture diagram, the axes, the values on a curve. Questions about a
figure's content then have nothing to match. A vision LLM can look at the
image crop (saved by rag.parse) and write that content down as searchable text.

What the model gets for each figure:
- the PNG crop (data/figures/<doc_id>/pXXX-figN.png)
- the section path and the printed caption
- nearby text: paragraphs that mention the figure ("Figure 5 shows...") first,
  then paragraphs from the same section, nearest first (capped at NEARBY_CHARS)

The context helps the model NAME things (what "MDQA" or "20 documents" means);
the prompt tells it to read every value off the image, not copy it from the text.

Results are cached in data/cache/captions.jsonl keyed by (model, image hash,
prompt), so the slow vision step runs once and chunking only reads the cache.

Usage (from rag-lab/; load the vision model only for this run, ~22 GB):
    lms unload --all && lms load qwen3.6-35b-a3b
    uv run python -m rag.caption                       # caption every figure
    uv run python -m rag.caption --limit 3 --show      # try a few first
    lms unload qwen3.6-35b-a3b
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from rag.paths import CAPTIONS_CACHE, parsed_dir

load_dotenv()
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.6-35b-a3b")
NEARBY_CHARS = 1500                 # context budget per figure
# Qwen3.6 is a thinking model. With reasoning on, busy line plots used all 2000
# tokens on reasoning and returned an EMPTY answer (finish_reason="length").
# Describing a figure needs no long reasoning: off is ~12 s/figure instead of 30-75 s.
# LM Studio honours reasoning_effort; chat_template_kwargs and "/no_think" were ignored.
REASONING_EFFORT = "none"
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FIG_NUM = re.compile(r"^\s*(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)

PROMPT = """You are describing a figure from a research paper so it can be found by search.

Paper: {title}
Section: {section}
Printed caption: {caption}
Nearby text from the paper (for naming things only):
{nearby}

Describe the figure in 3-5 sentences: what kind of figure it is (diagram, line plot, \
bar chart, table image, example...), what it shows, every labelled block or axis and \
legend entry, the key values or trends, and the main takeaway. Read labels and numbers \
from the image itself; do not copy numbers from the nearby text. Plain prose, no lists, \
no preamble."""


# ---------- pure helpers (tested in tests/test_caption.py) ----------

def figure_number(caption: str | None) -> int | None:
    """'Figure 5: The effect...' -> 5; None when the caption has no number."""
    m = _FIG_NUM.match(caption or "")
    return int(m.group(1)) if m else None


def mentions_figure(text: str, n: int) -> bool:
    """True if text refers to figure n ('Figure 5', 'Fig. 5', 'Figures 5'), not 'Figure 50'."""
    return re.search(rf"\b(?:Figures?|Figs?\.?)\s*{n}(?!\d)", text, re.IGNORECASE) is not None


def nearby_text(elements: list[dict], index: int, budget: int = NEARBY_CHARS) -> str:
    """Context for the figure at elements[index], most useful first, up to `budget` chars.

    1. paragraphs anywhere in the paper that mention this figure by number
    2. paragraphs in the same section, nearest to the figure first
    """
    fig = elements[index]
    is_text = lambda e: e["type"] in ("paragraph", "list_item") and e["text"]  # noqa: E731

    picked: list[int] = []
    if (n := figure_number(fig.get("caption"))) is not None:
        picked += [i for i, e in enumerate(elements) if is_text(e) and mentions_figure(e["text"], n)]

    same_section = [i for i, e in enumerate(elements)
                    if is_text(e) and e["section_path"] == fig["section_path"] and i not in picked]
    picked += sorted(same_section, key=lambda i: abs(i - index))

    parts, used = [], 0
    for i in picked:
        text = elements[i]["text"]
        if used + len(text) > budget:
            text = text[: max(budget - used, 0)]
        if text:
            parts.append(text)
            used += len(text)
        if used >= budget:
            break
    return "\n\n".join(parts) or "(none)"


def build_prompt(doc: dict, index: int) -> str:
    fig = doc["elements"][index]
    return PROMPT.format(
        title=doc["title"],
        section=" > ".join(fig["section_path"]),
        caption=fig.get("caption") or "(no printed caption)",
        nearby=nearby_text(doc["elements"], index),
    )


# ---------- cache ----------

class CaptionCache:
    """Append-only JSONL cache: {"key", "image_path", "model", "caption"} per line."""

    def __init__(self, path: Path = CAPTIONS_CACHE) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row["caption"]:  # never trust an empty caption (old bug: truncated replies)
                        self.data[row["key"]] = row["caption"]

    @staticmethod
    def key(model: str, image_bytes: bytes, prompt: str) -> str:
        settings = f"{model}|reasoning={REASONING_EFFORT}"
        raw = settings.encode() + b"|" + hashlib.sha256(image_bytes).digest() + b"|" + prompt.encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, image_path: str, model: str, caption: str) -> None:
        with self.lock:
            self.data[key] = caption
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                row = {"key": key, "image_path": image_path, "model": model, "caption": caption}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- the vision call ----------

def call_vision(image_bytes: bytes, prompt: str, model: str = VISION_MODEL) -> str:
    """One image + prompt to LM Studio's OpenAI-compatible API. Returns one line of prose."""
    from openai import OpenAI

    client = OpenAI(base_url=os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1"), api_key="lm-studio")
    b64 = base64.b64encode(image_bytes).decode()
    r = client.chat.completions.create(
        model=model, temperature=0,
        max_tokens=800,  # the answer is ~150-250 tokens; reasoning is off
        extra_body={"reasoning_effort": REASONING_EFFORT},
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    choice = r.choices[0]
    text = " ".join(_THINK.sub("", choice.message.content or "").split())  # one line
    if not text or choice.finish_reason == "length":
        # Fail loudly: caching an empty/cut-off caption would silently poison every later run.
        raise RuntimeError(f"vision model returned no complete caption "
                           f"(finish_reason={choice.finish_reason}, {len(text)} chars)")
    return text


class Captioner:
    """Looks up or creates the vision caption for a figure element."""

    def __init__(self, model: str = VISION_MODEL, offline: bool = False) -> None:
        self.model = model
        self.offline = offline  # True: cache only, never call the model (used by rag.chunk)
        self.cache = CaptionCache()
        self.stats = {"llm_calls": 0, "cache_hits": 0, "missing": 0, "seconds": 0.0}

    def caption(self, doc: dict, index: int) -> str | None:
        fig = doc["elements"][index]
        if not fig.get("image_path") or not Path(fig["image_path"]).exists():
            self.stats["missing"] += 1
            return None
        image_bytes = Path(fig["image_path"]).read_bytes()
        prompt = build_prompt(doc, index)
        key = self.cache.key(self.model, image_bytes, prompt)
        if (cached := self.cache.get(key)) is not None:
            self.stats["cache_hits"] += 1
            return cached
        if self.offline:
            self.stats["missing"] += 1
            return None
        start = time.perf_counter()
        text = call_vision(image_bytes, prompt, self.model)
        self.stats["seconds"] += time.perf_counter() - start
        self.stats["llm_calls"] += 1
        self.cache.put(key, fig["image_path"], self.model, text)
        return text


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Caption every figure with the vision LLM (cached).")
    ap.add_argument("--parser", default="docling", help="which parsed output to read")
    ap.add_argument("--model", default=VISION_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="caption only N figures in total")
    ap.add_argument("--show", action="store_true", help="print each caption")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    captioner = Captioner(args.model)
    done = 0
    for path in sorted(parsed_dir(args.parser).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for i, e in enumerate(doc["elements"]):
            if e["type"] != "figure" or (args.limit is not None and done >= args.limit):
                continue
            start = time.perf_counter()
            text = captioner.caption(doc, i)
            done += 1
            print(f"[{done}] {e.get('image_path')}  ({time.perf_counter() - start:.0f}s)", flush=True)
            if args.show:
                print(f"    printed: {e.get('caption')}\n    vision:  {text}\n")

    s = captioner.stats
    print(f"\n{done} figures: {s['llm_calls']} captioned, {s['cache_hits']} cached, "
          f"{s['missing']} missing image, {s['seconds']:.0f}s in the model -> {CAPTIONS_CACHE}")


if __name__ == "__main__":
    main()
