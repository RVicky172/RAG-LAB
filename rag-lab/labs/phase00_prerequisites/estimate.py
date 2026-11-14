# estimate.py
# Cloud cost estimate for a RAG workload:
#   cost = input_tokens * input_price + output_tokens * output_price

# Price per 1M tokens (input, output) in USD.
# EXAMPLE TIERS ONLY: replace with real numbers from the Bedrock pricing page (Phase 09).
PRICES = {
    "small model": (0.25, 1.25),
    "mid model":   (3.00, 15.00),
    "large model": (15.00, 75.00),
}

QUESTIONS = 10_000
INSTRUCTION_TOKENS = 300
ANSWER_TOKENS = 300


def tokens_per_question(n_chunks: float, chunk_tokens: int) -> tuple[int, int]:
    input_tokens = int(n_chunks * chunk_tokens) + INSTRUCTION_TOKENS
    return input_tokens, ANSWER_TOKENS


def cost(input_tokens: int, output_tokens: int, price: tuple[float, float]) -> float:
    in_price, out_price = price
    return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price


SCENARIOS = {
    "5 chunks x 500":   (5, 500),
    "halved (2.5x500)": (2.5, 500),   # same as 5 chunks x 250: 1,250 context tokens
}

for label, (n_chunks, chunk_tokens) in SCENARIOS.items():
    inp, out = tokens_per_question(n_chunks, chunk_tokens)
    total_in, total_out = inp * QUESTIONS, out * QUESTIONS
    print(f"\n{label}: {inp:,} in + {out} out per question "
          f"-> {total_in / 1e6:.1f}M in + {total_out / 1e6:.1f}M out for {QUESTIONS:,} questions")
    for model, price in PRICES.items():
        c = cost(total_in, total_out, price)
        in_share = cost(total_in, 0, price) / c
        print(f"  {model:<12} ${c:>9,.2f}   (input is {in_share:.0%} of the bill)")

# Savings from halving the context
full_in, _ = tokens_per_question(*SCENARIOS["5 chunks x 500"])
half_in, _ = tokens_per_question(*SCENARIOS["halved (2.5x500)"])
print("\nsavings from halving the chunks:")
for model, price in PRICES.items():
    a = cost(full_in * QUESTIONS, ANSWER_TOKENS * QUESTIONS, price)
    b = cost(half_in * QUESTIONS, ANSWER_TOKENS * QUESTIONS, price)
    print(f"  {model:<12} ${a - b:>9,.2f} saved ({(a - b) / a:.0%})")
