# Lab 1a notes: embeddings playground

Date: 2026-11-13

## Experiment 1: MiniLM vs nomic on 4 sentences (`compare.py`)

Sentences: (0) reset password, (1) forgot login, (2) quarterly revenue, (3) error code HX-4410.

| Pair | MiniLM (384 dims) | nomic-v1.5 (768 dims) |
|------|------------------:|----------------------:|
| 0–1 password vs login (same meaning, no shared words) | **0.68** | **0.73** |
| 0–2 password vs revenue (unrelated) | -0.01 | 0.44 |
| 2–3 revenue vs error code (unrelated) | 0.02 | 0.48 |
| 1–3 login vs error code (both "IT trouble") | 0.10 | 0.55 |

**What surprised me:**

1. **Both models match meaning without shared words.** "Reset my password" and "forgot my login credentials" share no keywords, yet both models rank them as the closest pair. That's semantic matching (lexical/BM25 would score ~0).
2. **nomic's scores live in a narrow band.** Unrelated pairs score 0.44–0.55, not ~0 like MiniLM. The gap between "related" and "unrelated" is ~0.63 for MiniLM but only ~0.25 for nomic.
   - **Lesson:** a similarity score means nothing on its own, and can't be compared across models. Only the *ranking* within one model matters.
   - This explains why my unanswerable golden questions still score 0.72–0.79: in nomic's range, that is "somewhat related", not "match". A fixed cut-off like "refuse below 0.7" won't work.
3. **The error code `HX-4410` is not understood as an identifier.** The models just see "error ... startup" and place it near the login sentence (IT trouble). Embeddings capture the *topic*, not the exact code. Searching for "HX-4410" would also match "HX-4411". That's why identifiers need keyword search (BM25, Phase 03).

## Experiment 2: silent truncation (`truncation_test.py`)

Two ~2,000-word texts (2,891 tokens), identical except for the last sentence.

| Model | Input limit | Similarity of the two texts | Read the ending? |
|-------|-------------|----------------------------:|------------------|
| MiniLM | 256 tokens | **1.0000** | No: 91% of the text ignored |
| nomic via LM Studio | **2,048** (as loaded) | **1.0000** | No |

**What surprised me:**

1. **No error, no warning from the embedding call.** Two texts with completely different endings got the exact same vector. (The tokenizer printed a warning for MiniLM, but the encode call itself returned normally.)
2. **nomic also truncated, even though it supports 8,192 tokens.** `lms ps` showed the embedding model loaded with **context 2048**, LM Studio's default. Testing different lengths:

   | Words | Similarity | |
   |------:|-----------:|---|
   | 300 | 0.9969 | ending read |
   | 1,000 | 0.9989 | ending read |
   | 1,300 | 0.9986 | ending read |
   | 1,500 | 1.0000 | **cut off** |
   | 2,000 | 1.0000 | **cut off** |

   So the cut-off is somewhere around 1,400 words (~2,048 tokens).
3. **Does it affect my current results?** No. My chunks are 800 characters (~200 tokens), far below 2,048. It **will** matter if Phase 02 tries large chunks or whole-page/whole-section embeddings.

**Fix:** reload the embedding model with a bigger context, e.g.

```bash
lms unload text-embedding-nomic-embed-text-v1.5
lms load text-embedding-nomic-embed-text-v1.5 --context-length 8192
```

Then re-run `truncation_test.py`: the 2,000-word case should drop below 1.0000.

## Takeaways for choosing an embedding model

- Max input tokens must be **≥ chunk size**, and check what the *server* is configured to, not just the model card.
- Scores are model-specific; compare rankings, not raw numbers.
- Embeddings handle paraphrases well and identifiers badly, so plan for hybrid search.
