# truncation_test.py: Lab 1a extra
# Shows that an embedding model silently ignores text beyond its input limit.
#
# Experiment: two ~2,000-word texts that are IDENTICAL except for the last sentence.
#   - If the model reads the whole text, the two vectors differ (similarity < 1).
#   - If the model cuts the text off early, it never sees the last sentence,
#     so both vectors are IDENTICAL (similarity = 1.000). No error, no warning.
#
# Run from rag-lab/ (LM Studio must be running for the nomic part):
#     uv run python labs/phase01_naive_rag/lab1a_embeddings/truncation_test.py

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

PDF_PATH = "data/raw/1706.03762v7.pdf"
TARGET_WORD_COUNT = 2000


# ---------- Build two long texts that differ only at the very end ----------

def first_words_of_pdf(pdf_path, word_count):
    all_words = []
    for page in PdfReader(pdf_path).pages:
        all_words += (page.extract_text() or "").split()
        if len(all_words) >= word_count:
            break
    return " ".join(all_words[:word_count])


long_text = first_words_of_pdf(PDF_PATH, TARGET_WORD_COUNT)
text_ending_a = long_text + " The secret maintenance code is HX-4410 and it opens the north door."
text_ending_b = long_text + " Bananas are yellow and monkeys like to eat them in the afternoon."


def cosine_similarity(vector_a, vector_b):
    return float(vector_a @ vector_b / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))


# ---------- 1. MiniLM: small model, 256-token limit ----------

minilm = SentenceTransformer("all-MiniLM-L6-v2")
token_count = len(minilm.tokenizer(long_text)["input_ids"])

print("=== all-MiniLM-L6-v2 ===")
print(f"Text length:            {len(long_text.split())} words = {token_count} tokens")
print(f"Model input limit:      {minilm.max_seq_length} tokens")
print(f"Tokens silently ignored: {max(token_count - minilm.max_seq_length, 0)} "
      f"({max(token_count - minilm.max_seq_length, 0) / token_count:.0%} of the text)")

minilm_vector_a, minilm_vector_b = minilm.encode([text_ending_a, text_ending_b])
print(f"Similarity of the two texts: {cosine_similarity(minilm_vector_a, minilm_vector_b):.4f}"
      "   <- 1.0000 means the different endings were never read")


# ---------- 2. nomic via LM Studio: 8,192-token limit ----------

print("\n=== nomic-embed-text-v1.5 (LM Studio) ===")
lm_studio = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")
try:
    response = lm_studio.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=["search_document: " + text_ending_a, "search_document: " + text_ending_b],
    )
    nomic_vector_a = np.array(response.data[0].embedding)
    nomic_vector_b = np.array(response.data[1].embedding)
    print(f"Similarity of the two texts: {cosine_similarity(nomic_vector_a, nomic_vector_b):.4f}"
          "   <- below 1 means the model read the ending")
except Exception as error:
    print(f"LM Studio refused the long input: {error}")
    print("Check the embedding model's context length in LM Studio.")


# ---------- What to take away ----------
# - Chunks longer than the embedding model's input limit lose their tail SILENTLY.
# - "Max input tokens" must be >= your chunk size. Check it when choosing a model.
# - A fact at the end of a long chunk can be invisible to search.
