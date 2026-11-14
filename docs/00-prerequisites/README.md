# Phase 00: Prerequisites & Local AI Setup

**Goal:** Be fluent with the tools and vocabulary before the theory starts, with a working **local** model stack in LM Studio. No API key needed.
**Time:** about 1 week part-time.
**Read first:** [MODELS.md](../MODELS.md) explains which model does what and why.

---

## Why this phase matters

Almost every RAG design decision is a trade-off measured in **tokens**: chunk size versus the embedding model's input limit, how many chunks fit in the context window, and how long (locally) or how much money (in the cloud) each query takes. If tokens and context windows are fuzzy, every later decision is a guess.

Running models locally makes these limits *visible*: you set the context length yourself, you watch memory use, and you see tokens per second.

---

## Learning steps (in order)

### Step 1: Python environment

- Tools: **uv** (Python and package manager), **Python 3.12** for the project, VS Code with the Python and Jupyter extensions.
- Know these Python features: list and dict comprehensions, dataclasses or pydantic models, generators (`yield`), type hints, reading and writing JSONL, `concurrent.futures.ThreadPoolExecutor`.

#### Why uv (and not plain `pip` + `venv`)

| | `python -m venv` + `pip` | **uv** (recommended) |
|---|---|---|
| Speed | Slow downloads, slow dependency resolution | 10–100× faster; global cache, so a second project reuses downloads |
| Python versions | Uses whatever Python is installed (3.14 here) | Installs and pins the version you want (`uv python install 3.12`) |
| Activation needed? | Yes, and it's easy to get wrong (see below) | No: `uv run <cmd>` always uses the project's `.venv` |
| Reproducibility | Hand-written `requirements.txt` | `pyproject.toml` + exact versions in `uv.lock` |
| CUDA PyTorch | Remember `--index-url` every time | Configured once in `pyproject.toml` |

**Why Python 3.12?** This machine has Python **3.14** (system) and **3.15** (uv-managed) installed. PyTorch only ships wheels for specific Python versions (currently up to 3.14), and later-phase libraries (FAISS, FlagEmbedding, Docling, ColPali, GraphRAG tooling) often lag further behind. **3.12** is the safe choice for the whole ML stack. uv installs it alongside the others without touching your system Python.

#### Current state of `rag-lab/` (already done)

| File | Content | Purpose |
|------|---------|---------|
| `.python-version` | `3.12` | Tells uv which Python to use in this folder |
| `pyproject.toml` | `requires-python = ">=3.12,<3.13"`, the cu128 index, all base packages | Project definition |
| `uv.lock` | Exact versions, incl. `torch 2.11.0+cu128` for Python 3.12 | Reproducible installs |
| `.venv/` | Created by `uv sync` (git-ignored) | The environment itself |
| `.env` | LM Studio URL + model names (git-ignored) | Model configuration for all labs |

**✅ Verified on this machine:** `3.12.13  2.11.0+cu128  True  NVIDIA GeForce RTX 5080`

| Package | Installed version |
|---------|-------------------|
| torch (CUDA 12.8) | 2.11.0+cu128 |
| openai | 3.50.0 |
| sentence-transformers | 6.3.0 |
| transformers | 5.22.2 |
| numpy · pandas | 2.5.4 · 3.1.1 |
| pydantic · python-dotenv · tqdm | 2.14.3 · 1.2.4 · 4.70.1 |
| pytest · ipykernel (dev) | 9.2.0 · 7.4.0 |

The environment is complete for Phases 00–01. Later phases add their own libraries with `uv add` when they start (e.g. `pypdf`, `faiss-cpu`, `rank-bm25`, `FlagEmbedding`, `docling`, `fastapi uvicorn`). Steps 1–5 below show how the project was set up, so you can repeat it for a new project or on a new machine (there, just run `uv sync`).

#### Setup with uv, step by step (Git Bash; the same commands work in PowerShell)

**1. Create the project on Python 3.12**

```bash
cd /d/CodeBase/AI/RAG/rag-lab
rm -rf .venv                        # remove any old venv (it may be on the wrong Python)
uv python install 3.12              # a managed Python 3.12 (no-op if already installed)
uv init --bare --python 3.12        # creates pyproject.toml only (--bare does NOT write .python-version)
uv python pin 3.12                  # writes .python-version, so uv always uses 3.12 here
```

**2. Stop the Python version drifting upward.** In `pyproject.toml`, set:

```toml
requires-python = ">=3.12,<3.13"
```

> **Why both the pin and the upper bound?** With `requires-python = ">=3.12"` and no `.python-version`, uv picks the **newest** installed Python (3.15 here). `uv add torch` then fails with *"`torch==…+cu128` can't be installed because it doesn't have a source distribution or wheel for the current platform … You're using CPython 3.15 (`cp315`)"*. The pin selects 3.12; the upper bound makes the lock file valid only for 3.12.

**3. Point torch at the CUDA build** (append to `pyproject.toml` **before** adding packages). Otherwise `sentence-transformers` pulls the **CPU-only** torch from PyPI and your RTX 5080 sits idle:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"   # CUDA 12.8+ build: required for RTX 50-series (Blackwell)
explicit = true                                  # use this index ONLY for packages mapped below

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

**4. Install** (torch is ~3 GB, so the first sync takes a while; later projects reuse uv's cache):

```bash
uv add torch                         # adds torch to pyproject.toml + uv.lock, then installs
                                     # (already in this project: run `uv sync` instead)
uv add openai numpy pandas python-dotenv pydantic tqdm sentence-transformers transformers
uv add --dev pytest ipykernel        # dev-only tools
```

**5. Verify** (all three values must match):

```bash
uv run python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 3.12.13 2.11.0+cu128 True NVIDIA GeForce RTX 5080
```

| Check | Good | If wrong |
|-------|------|----------|
| Python version | `3.12.x` | `.python-version` missing → `uv python pin 3.12`, then `rm -rf .venv && uv sync` |
| torch version | ends in `+cu128` | ends in `+cpu` or has no suffix → the index config (step 3) is missing; fix it, then `uv sync` |
| `cuda.is_available()` | `True` | Update the NVIDIA driver; check that the version ends in `+cu128` |

#### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `uv add torch` → "…doesn't have a … wheel for the current platform … `cp315`" | uv chose Python 3.15 | Steps 1–2: `uv python pin 3.12`, set `requires-python = ">=3.12,<3.13"`, `rm -rf .venv` |
| `Using CPython 3.15.x` / `3.14.x` printed by uv | No pin, or an old `.venv` | `uv python pin 3.12 && rm -rf .venv && uv sync` |
| `torch.__version__` ends in `+cpu` | torch came from PyPI, not the CUDA index | Add the step 3 config, then `uv lock --upgrade-package torch && uv sync` |
| `bash: .venvScriptsactivate: command not found` | Backslashes in Git Bash | Use `uv run …`, or `source .venv/Scripts/activate` |
| pip says `Requirement already satisfied … pythoncore-3.14-64\Lib\site-packages` | The venv wasn't activated, so pip used the global Python | Use `uv add` / `uv run` instead of pip |
| `uv sync` is very slow | First torch download (~3 GB) | One-time cost; let it finish. uv caches it for future projects. |
| Warning that a GPU "sm_120 is not compatible" | A torch build older than CUDA 12.8 | Make sure the version ends in `+cu128` (or newer CUDA) |

Daily use, with no activation needed:

```bash
uv run python hello_llm.py          # run a script inside the project env
uv run pytest                       # run tests
uv add pypdf                        # add a package later (updates pyproject.toml + uv.lock)
uv sync                             # recreate .venv exactly from uv.lock (new machine, after git pull)
```

In VS Code: Ctrl+Shift+P → *Python: Select Interpreter* → `rag-lab/.venv/Scripts/python.exe`. Notebooks then use the same environment (that's why `ipykernel` is installed).

#### If you prefer to activate the venv: use the right command for your shell

| Shell | Activate command |
|-------|------------------|
| **Git Bash** (MINGW64) | `source .venv/Scripts/activate` (forward slashes, and **`source`**) |
| PowerShell | `.venv\Scripts\Activate.ps1` |
| Command Prompt (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Common mistakes in Git Bash:
- `.venv\Scripts\activate`: Bash treats `\` as an escape character, so the path becomes `.venvScriptsactivate` → "command not found".
- `.venv/Scripts/activate` without `source`: the script runs in a **child process** and exits, so your shell is **not** activated. pip then installs into the **global** Python. The tell-tale sign is pip output saying `Requirement already satisfied: ... in C:\Users\...\pythoncore-3.14-64\Lib\site-packages`.
- Check before installing: `which python` should point inside `rag-lab/.venv/`. With uv you skip this problem entirely.

> **uv vs uvicorn:** different tools with similar names. **uv** manages Python versions, environments and packages (this step). **uvicorn** is a *web server* that runs FastAPI apps. You'll use it in Phase 08 (`uv add fastapi uvicorn`, then `uv run uvicorn api.main:app --reload`). It doesn't install packages.

> Why the `openai` package with no OpenAI account? LM Studio exposes an **OpenAI-compatible API** on your machine. The `openai` library is just a convenient client for it; nothing leaves your computer.

### Step 2: LM Studio setup

**Current status** (`lms ls` on this machine; full list in [MODELS.md §0](../MODELS.md#0-final-model-checklist-what-to-download)):

| Model | LM Studio name | Status |
|-------|----------------|--------|
| Qwen3-30B-A3B-Instruct-2507 | `qwen/qwen3-30b-a3b-2507` | ✅ |
| Qwen3.6-35B-A3B (+ vision) | `qwen3.6-35b-a3b` | ✅ |
| Qwen3-4B-Instruct-2507 | `qwen/qwen3-4b-2507` | ✅ |
| gpt-oss-20b | `openai/gpt-oss-20b` | ✅ |
| nomic-embed-text-v1.5 | `text-embedding-nomic-embed-text-v1.5` | ✅ (built in) |
| Qwen3-Embedding-0.6B | `text-embedding-qwen3-embedding-0.6b` | ✅ |
| Qwen3-Embedding-4B (optional) | `text-embedding-qwen3-embedding-4b` | ✅ |

All 7 models are installed (58.45 GB). Nothing left to download.

1. Open the LM Studio app (the `lms` CLI needs the app or its server running). If `lms` isn't on your PATH, run `~/.lmstudio/bin/lms bootstrap` once.
2. Developer tab → **Start Server** (or `lms server start`). The API is at `http://localhost:1234/v1`.
3. Load the generator with a **context length of 16,384–32,768** (the default is often only 4,096, which silently truncates RAG prompts):
   ```bash
   lms ls                                                   # exact model identifiers
   lms load qwen/qwen3-30b-a3b-2507 --context-length 32768
   lms load text-embedding-nomic-embed-text-v1.5
   lms ps                                                   # what's loaded now
   ```
   In the app's load settings, also look at **GPU offload** and (in recent versions) **keeping MoE expert weights on the CPU**. With 16 GB of VRAM and 127 GB of RAM, that's how the 30B MoE model runs fast.
4. Check: `curl http://localhost:1234/v1/models`.

`rag-lab/.env` (git-ignored) ✅ already created with these values:

```ini
# Model names exactly as `lms ls` reports them on this machine
LLM_PROVIDER=lmstudio
LMSTUDIO_URL=http://localhost:1234/v1
GEN_MODEL=qwen/qwen3-30b-a3b-2507
HELPER_MODEL=qwen/qwen3-4b-2507
VISION_MODEL=qwen3.6-35b-a3b
JUDGE_MODEL=openai/gpt-oss-20b
EMBED_PROVIDER=lmstudio
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
# From Phase 02: text-embedding-qwen3-embedding-0.6b
# Phase 02-03 size comparison: text-embedding-qwen3-embedding-4b
```

Add `.env` and `.venv/` to `rag-lab/.gitignore`. **Do** commit `pyproject.toml`, `uv.lock` and `.python-version`: they let anyone (or future you) recreate the exact environment with `uv sync`.

### Step 3: How LLMs work (just enough)

| Concept | What you must be able to say |
|---------|------------------------------|
| Token | A piece of text (~4 characters of English). Models read, generate and (in the cloud) bill in tokens. |
| Context window | Maximum tokens (input + output) the model attends to in one request. Locally, **you** choose it at load time, and bigger costs more memory (KV cache). |
| Next-token prediction | The model generates one token at a time, based on everything before it. |
| Parametric knowledge | What the model "knows" from training, frozen at a cutoff date. |
| Hallucination | Fluent output not supported by facts. The model optimizes for plausible text, not truth. |
| System prompt | Instructions that set behaviour for the whole conversation. |
| Temperature | Randomness in token choice. RAG answers: 0.1–0.3. JSON or helper tasks: 0. |
| Streaming | Receiving tokens as they're generated, for a lower *time to first token*. |
| Quantization | Storing weights in fewer bits (Q4_K_M ≈ 4.5 bits) to fit models in memory, with a small quality cost. |
| Mixture of Experts (MoE) | Only some "expert" sub-networks run per token. Qwen3-30B-A3B uses ~3B of its 30B parameters per token, so it's fast. |
| Thinking / reasoning models | Some models write a hidden `<think>...</think>` section before answering. Strip it before showing or parsing the answer. |

Resources: Andrej Karpathy's *Intro to Large Language Models* and *Let's build the GPT Tokenizer* (YouTube); Hugging Face's *Quantization* docs; the Qwen3 model cards.

### Step 4: Your first local LLM call

```python
# hello_llm.py
import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")  # key is ignored locally

t0 = time.perf_counter()
resp = client.chat.completions.create(
    model=os.getenv("GEN_MODEL"),
    messages=[{"role": "system", "content": "You are a concise assistant."},
              {"role": "user", "content": "Explain RAG in two sentences."}],
    temperature=0.2, max_tokens=300,
)
dt = time.perf_counter() - t0
print(resp.choices[0].message.content)
u = resp.usage
print(f"prompt={u.prompt_tokens} completion={u.completion_tokens} "
      f"time={dt:.1f}s  ~{u.completion_tokens/dt:.0f} tok/s")
```

Also try **streaming** (`stream=True`, then iterate over the chunks) and notice how time to first token differs from total time.

Exercises:
1. Ask about something that happened last week. Observe the knowledge cutoff.
2. Ask about your own company's internal policy. Observe that it invents an answer or declines.
3. Paste a policy paragraph into the prompt and ask again. Observe the difference. **That is RAG in miniature.**
4. Run the same question on **Qwen3-4B** and **Qwen3-30B-A3B**. Compare quality and speed. This is the helper vs generator trade-off.
5. Load the generator with context 4,096, then paste a 6,000-token document. See what happens. That's why you set context length deliberately.

### Step 5: Tokens, latency and cost intuition

- **Count tokens** with the model's own tokenizer (only the tokenizer files download, not the model):
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
  print(len(tok.encode("Employees accrue 1 day of sick leave per month.")))
  ```
  Different model families use different tokenizers, so counts differ. Also check `resp.usage`.
- **Local cost = time.** Build `bench.py`: for prompts with 1K, 4K, 16K tokens of context, measure time to first token, total time and tokens/second. Longer prompts take longer to *read* (prompt processing), not just to answer.
- **Cloud cost estimate:** `cost = input_tokens × input_price + output_tokens × output_price`. Build `cost.py` that takes the token counts from your local runs and a price table (fill it from the Bedrock pricing page in Phase 09).
- Exercise: estimate the cost of 10,000 questions with 5 chunks × 500 tokens + 300 tokens of instructions + a 300-token answer. Then halve the chunks and recompute. This is why rerankers and context compression exist (Phase 05).

### Step 6: Vectors and linear algebra (just enough)

- A vector is a list of numbers; an embedding is a vector that represents meaning.
- Dot product, vector length (norm), normalization to unit length.
- Cosine similarity = dot product of two unit-length vectors.
- Matrix multiplication as "compare one query against N documents at once".

```python
import numpy as np
a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.5])
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(round(cos, 4))   # close to 1.0 → pointing the same direction
```

First local embedding call:

```python
emb = client.embeddings.create(model=os.getenv("EMBED_MODEL"),
                               input=["search_query: reset my password",
                                      "search_document: How to recover your login credentials"])
v = [np.array(d.embedding) for d in emb.data]
print(len(v[0]), float(v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))))
```

(nomic-embed expects the `search_query:` / `search_document:` prefixes; Phase 01 explains why.)

Resource: 3Blue1Brown, *Essence of Linear Algebra* (chapters 1–3, 9).

### Step 7: Prompting basics for RAG

- Give clear roles and rules: "Answer only from the sources."
- Put long documents at the **top** of the prompt and the question at the **bottom**.
- Use XML-style tags to separate data from instructions: `<sources>...</sources>`.
- Ask for citations in a fixed format, such as `[1]`.
- Define an explicit refusal: *"If the sources don't contain the answer, say exactly: I couldn't find this in the documents."*
- Smaller local models need **clearer and shorter** instructions than frontier models. Test your prompt on the 4B model too.

### Step 8: Git and project hygiene

- Init a git repo for `rag-lab/`. Commit after every milestone.
- Never commit `.env`, keys (AWS later), or private documents.
- Keep a `notes.md` per phase where you explain concepts in your own words.

---

## Milestone

A `rag-lab/` repo containing:
- `hello_llm.py`: a local chat call (normal and streaming) printing answer, tokens and tokens/second
- `bench.py`: time to first token and total latency for 1K / 4K / 16K-token prompts on the 4B and 30B models
- `cost.py`: estimated cloud cost from token counts
- `notes.md` with your own definitions of: token, context window, quantization, MoE, hallucination, embedding, cosine similarity

## Exit checklist

- [ ] LM Studio server runs; I can list models via `/v1/models`.
- [ ] I loaded the generator with a deliberate context length and know why.
- [ ] I can call a local LLM and a local embedding model from Python.
- [ ] I can explain why the model can't answer questions about my private documents.
- [ ] I can explain quantization and why MoE models are fast on my machine.
- [ ] I know my local tokens/second and how prompt length affects latency.
- [ ] I can compute cosine similarity by hand and in NumPy.

## Next

→ [Phase 01: Foundations & the Naive RAG Pipeline](../01-foundations-naive-rag/README.md)
