# prompting.py
# A grounded RAG prompt, tested on the main model and the 4B helper.
#
# Rules this file applies:
#   1. Clear role and rules: "Answer only from the sources."
#   2. Long documents at the top, the question at the bottom.
#   3. XML-style tags separate data from instructions: <sources>...</sources>.
#   4. Citations in a fixed format: [1], [2].
#   5. An exact refusal string when the sources don't contain the answer.
#   6. Short, plain instructions so a 4B model follows them too.
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")  # key is ignored locally

MODELS = {
    "main": os.getenv("GEN_MODEL", "qwen/qwen3-30b-a3b-2507"),
    "4b": os.getenv("HELPER_MODEL", "qwen/qwen3-4b-2507"),
}

REFUSAL = "I couldn't find this in the documents."

# Rule 1 + 6: one role, few short rules, no hedging words.
SYSTEM = "You answer questions using only the sources you are given."

RULES = f"""Rules:
- Answer only from the sources above.
- Cite every fact with its source number in square brackets, like [1].
- If the sources don't contain the answer, say exactly: {REFUSAL}
- Keep the answer short."""


def format_sources(sources: list[str]) -> str:
    """Rule 3: wrap each document in tags with a stable id the model can cite."""
    body = "\n".join(
        f'<source id="{i}">\n{text.strip()}\n</source>' for i, text in enumerate(sources, 1)
    )
    return f"<sources>\n{body}\n</sources>"


def build_messages(question: str, sources: list[str]) -> list[dict]:
    """Rule 2: sources first, rules next, question last."""
    user = f"{format_sources(sources)}\n\n{RULES}\n\n<question>{question.strip()}</question>"
    print([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ask(question: str, sources: list[str], model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(question, sources),
        temperature=0.0,  # deterministic, so prompt changes are easy to compare
        max_tokens=300,
    )
    print({resp.choices[0].message.content.strip()})
    return resp.choices[0].message.content.strip()


def check(answer: str, n_sources: int, should_refuse: bool) -> tuple[bool, str]:
    """Check the answer follows the rules: exact refusal, or valid [n] citations."""
    if should_refuse:
        return (answer == REFUSAL, "exact refusal" if answer == REFUSAL else "expected refusal")
    if REFUSAL in answer:
        return False, "refused an answerable question"
    cites = [int(c) for c in re.findall(r"\[(\d+)\]", answer)]
    if not cites:
        return False, "no citations"
    bad = [c for c in cites if not 1 <= c <= n_sources]
    if bad:
        return False, f"cites missing sources {bad}"
    return True, f"cites {sorted(set(cites))}"


SOURCES = [
    "The Aurora X2 laptop has a 14-inch OLED display and weighs 1.2 kg. "
    "It ships with 32 GB of RAM.",
    "Aurora X2 battery life is rated at 18 hours for video playback. "
    "The charger supports 100 W USB-C fast charging.",
    "The Aurora warranty covers manufacturing defects for 2 years. "
    "Accidental damage is not covered.",
]

CASES = [
    # (question, should_refuse)
    ("How much does the Aurora X2 weigh?", False),
    ("What is the battery life and how fast does it charge?", False),
    ("Is accidental damage covered by the warranty?", False),
    ("What processor does the Aurora X2 use?", True),   # not in the sources
    ("How much does the Aurora X2 cost?", True),        # not in the sources
]


def main() -> None:
    for label, model in MODELS.items():
        print(f"\n=== {label}: {model} ===")
        passed = 0
        for question, should_refuse in CASES:
            t0 = time.perf_counter()
            answer = ask(question, SOURCES, model)
            dt = time.perf_counter() - t0
            ok, why = check(answer, len(SOURCES), should_refuse)
            passed += ok
            print(f"\n[{'PASS' if ok else 'FAIL'}] {question}  ({why}, {dt:.1f}s)")
            print(f"  {answer}")
        print(f"\n{label}: {passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
