# token_count.py
# Count tokens with each model's own tokenizer (only tokenizer files download, not model weights).
from transformers import AutoTokenizer

GEN_TOKENIZER = "Qwen/Qwen3-30B-A3B-Instruct-2507"   # generator (chat model)
EMB_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"      # embedding model (BERT-style tokenizer)

SAMPLES = {
    "policy sentence": "Employees accrue 1 day of sick leave per month.",
    "numbers/dates":   "Invoice #INV-2026-00417 dated 13/11/2026 totals $12,345.67 (incl. 18% GST).",
    "code":            "def chunk(text, size=500, overlap=50):\n    return [text[i:i+size] for i in range(0, len(text), size-overlap)]",
    "hindi":           "कर्मचारियों को हर महीने एक दिन की बीमारी की छुट्टी मिलती है।",
    "long paragraph":  ("Retrieval-augmented generation retrieves relevant passages from a knowledge base "
                        "and adds them to the prompt, so the model answers from those sources instead of "
                        "relying only on what it memorised during training. ") * 5,
}

gen_tok = AutoTokenizer.from_pretrained(GEN_TOKENIZER)
emb_tok = AutoTokenizer.from_pretrained(EMB_TOKENIZER)


def count(tok, text: str) -> int:
    # add_special_tokens=False: count only the text itself, not [CLS]/[SEP] etc.
    return len(tok.encode(text, add_special_tokens=False))


# 1. Token counts and characters-per-token for different kinds of text
print(f"{'sample':<17}{'chars':>7}{'qwen':>7}{'nomic':>7}{'chars/tok (qwen)':>18}")
for name, text in SAMPLES.items():
    q, n = count(gen_tok, text), count(emb_tok, text)
    print(f"{name:<17}{len(text):>7}{q:>7}{n:>7}{len(text) / q:>18.2f}")

# 2. See how a sentence is actually split
text = SAMPLES["numbers/dates"]
ids = gen_tok.encode(text, add_special_tokens=False)
print("\nQwen pieces:", [gen_tok.decode([i]) for i in ids])

# 3. Chat-template overhead: the prompt the server really sees is bigger than your text
messages = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain RAG in two sentences."}]
raw = sum(count(gen_tok, m["content"]) for m in messages)
prompt_text = gen_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
templated = count(gen_tok, prompt_text)
print("\nwhat the model really sees:\n" + prompt_text)
print(f"\nchat messages: content={raw} tokens, with chat template={templated} tokens "
      f"(overhead {templated - raw}). Compare with resp.usage.prompt_tokens in hello_llm.py.")

# 4. Will a chunk fit the embedding model? (nomic-embed-text-v1.5 accepts up to 8192 tokens)
chunk = "search_document: " + SAMPLES["long paragraph"]   # nomic needs a task prefix
print(f"\nembedding chunk: {count(emb_tok, chunk)} tokens (limit 8192)")
