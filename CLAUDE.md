# CLAUDE.md

Guidance for Claude working in this repository. Read this first, then [README.md](README.md) (roadmap + progress tracker) and [rag-lab/README.md](rag-lab/README.md) (code map).

## What this project is

A personal **RAG learning plan**: phases 00–10 in `docs/NN-*/README.md`, built hands-on in `rag-lab/`. Local-first (LM Studio + open models on an RTX 5080, 16 GB VRAM), AWS Bedrock only in Phase 09. The user is learning: the goal is understanding, not just working code.

## How to work with the user

- **Follow the phase README step by step, in order.** Do the current step, then say what the next one is. Don't jump ahead.
- **Explain the *why*** in plain language: what the step is for, what the output means, what went wrong and why. Correct misunderstandings directly (e.g. "no parser reads pixels").
- **Keep the user in the loop on observations:** they inspect outputs (parser files, chunks) and fill in notes; help them fill `defects.md` / `shootout.md` / `results.md` with concrete evidence (file, page, line).
- **Measure, don't guess.** "Change one thing, measure, keep it or revert it." Every pipeline change ends with an eval row in `results.md`. Keep baselines (e.g. the pypdf parser) for one-variable comparisons.
- Write clean, commented, typed code; small modules; tests for pure helpers.

## Where things go (rag-lab/)

- `src/rag/`: the library (installed package `rag`). Reusable pipeline code only.
- `labs/phaseNN_<topic>/labNx_<name>/`: one-off learning scripts and their notes.
- `eval/phaseNN_<topic>/`: eval scripts, `results.md`, raw `runs.jsonl`; shared `eval/eval_core.py`, `eval/golden.jsonl`; summary `eval/results.md` (one row per phase).
- `data/`: git-ignored (except `data/README.md`), in pipeline order: `raw/ → parsed/<parser>/ → figures/ → chunks/ → cache/ → indexes/<phase>/ → lab_outputs/<phase>/<lab>/`.
- **Never hard-code a data path in pipeline/eval code: import it from `rag.paths`.** (Phase 01 learning scripts still use literal `data/indexes/phase01/...` paths; that's fine.)

## Commands (run from `rag-lab/`, always via `uv run`)

```bash
uv run python -m rag.parse data/raw [--parser pypdf] [--force]
uv run python -m rag.chunk {recursive|structure|semantic|contextual} [--size 512 --overlap 64] [--parser pypdf]
uv run python -m eval.phase02_chunking.eval_numpy [config ...] [--verbose]   # also eval_faiss, eval_chroma
uv run python -m eval.phase01_naive.eval_numpy
uv run pytest
uv add <pkg>        # never pip
```

## Environment gotchas (learned the hard way)

- **Windows + Git Bash.** Set `sys.stdout.reconfigure(encoding="utf-8")` in CLIs (console is cp1252); use `PYTHONIOENCODING=utf-8` for one-off scripts.
- **CUDA torch:** `torch` and `torchvision` both come from the `pytorch-cu128` index (`[tool.uv.sources]` in `pyproject.toml`). A new package pulling torchvision from PyPI gives a CPU build. Check `torch.__version__` / `torchvision.__version__` end in `+cu128`.
- **LM Studio:** start with `lms server start`; check `lms ps` and `nvidia-smi`. **Load only what the step needs:** idle big models (vision 35B = 22 GB) fill VRAM and make the 4B helper ~12× slower. Unload with `lms unload <id>`.
- **LM Studio parallel requests share one context (KV cache):** 4 parallel × 10K-token prompts overflow a 29K context → HTTP 400. Contextual retrieval runs with `--workers 1`.
- **Docling:** OCR off (`do_ocr=False`, digital PDFs); first run downloads models from Hugging Face and the connection sometimes resets: just retry. Its `section_header` levels are all 1: `rag.parse` derives levels from numbering.
- **Long jobs** (contextual retrieval ≈ 11 chunks/min ≈ 16 min for the corpus): run in the background; results are cached in `data/cache/contexts.jsonl` (keyed by model + chunk hash), so re-runs are free.
- `.gitignore` does **not** allow trailing comments on a pattern line.
- **Vision model (Qwen3.6):** load with `lms load qwen3.6-35b-a3b --context-length 8192 --parallel 1` (default 262K x 4 fills VRAM, HTTP 400). It is a thinking model: send `reasoning_effort: "none"` or it can spend all output tokens reasoning and return an empty answer.

## Current state (update this section at the end of each session)

- **Corpus:** 4 arXiv PDFs in `rag-lab/data/raw/` (1706.03762 Attention, 2005.11401 RAG, 2307.03172 Lost in the Middle, 2309.15217 RAGAS; the last two are two-column).
- **Golden set:** v4, 57 questions (53 answerable) in `rag-lab/eval/golden.jsonl`: q001–q021 first two papers, q022–q042 two-column papers, q043–q057 `chart` questions (answer read off a figure; RAGAS has no figures). Page-level hit@5 overstates chart performance: use **fig@5** (figure chunk itself in top 5; added to eval_core).
- **Phase 00, 01, 02 done.** Phase 02 decision: Docling + structure-aware **400/64** with section path + **vision captions** (golden v4: hit@5 0.92, MRR 0.80, chart fig@5 0.93; config `docling-structure-400-64-vcap`; without captions 0.91 / 0.76 / 0.53). Contextual retrieval inconsistent (+0.05 MRR at 512, −0.06 at 400) → not adopted until BM25/reranker. Semantic dropped. Misses: q013, q021 (paraphrase → Phases 03–04), q023 (topic-word trap → reranker, Phase 05), q048 (vision description dropped the figure's example text).
- Phase 02 Steps 5–6: vision captions (`rag.caption`, cache `data/cache/captions.jsonl`). Embedding: nomic kept; Qwen3-4B ties (0.91 / 0.80) but solves q013/q021/q023 → try as a 2nd retriever in Phase 03; Qwen3-0.6B 0.85 / 0.74. Qwen3 needs the model card's query instruction verbatim (fixed in `rag/embed.py`, was −15 points). Chunk size: 256–512 flat, 1024 worse → 400 kept.
- **Next: Phase 03** (vector search & indexing: ANN benchmark, BM25 + hybrid with RRF). Ideas carried over: Qwen3-Embedding-4B as a second retriever; retry contextual retrieval with BM25; optional q048 "transcribe visible text" caption prompt.
- **Git:** remote `rag-lab` → https://github.com/RVicky172/RAG-LAB.git (`main` tracks it). `origin` → RAG-Practice (older, not updated). Commit only when asked; end messages with the Co-Authored-By line.
