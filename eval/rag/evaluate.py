"""
Evaluate the RAG pipeline (Doc 1) on the questions in `eval/rag/questions.jsonl`.

For each question we run `answer()` and then measure 4 things:

    retrieval_hit    Did search find a chunk from the right PDF?        (plain code, no LLM)
    context_recall   Do the found chunks contain the facts we need?     (LLM judge)
    faithfulness     Is the answer backed by the chunks, not invented?  (LLM judge)
    answer_accuracy  Does the answer match the reference answer?        (LLM judge)

Why these four? When a question fails, they show where it went wrong:
    - low hit / recall      -> retrieval is the problem (wrong chunks found)
    - low faithfulness      -> the LLM made things up
    - low accuracy only     -> the LLM misread good chunks, or the judge is wrong

An "LLM judge" is an LLM that grades another LLM's work. RAGAS provides the
judge prompts. Judges are noisy, so read the failing rows yourself.

Run: uv run python -m eval.rag.evaluate
Output: a START and a DONE line per question (with timestamps, so you can see
WORKERS questions running in parallel), the token use of each question, the
averages, and eval/rag/results/ with results.json and report.html (open it in a browser).
Every question is also one trace in Phoenix (project "rag-eval"): the answer call
and all judge calls, with their tokens. The report links to each one.

The LangGraph agent eval (Doc 4) reuses main() below with its own system and names.
"""

import asyncio
import sys
import time
from pathlib import Path

from ragas.metrics.collections import AnswerAccuracy, ContextRecall, Faithfulness

from app.rag.query import answer
from app.shared.tracing import current_trace_id, start_tracing
from eval.rag.report import write_rag_report
from eval.rag.tokens import answer_tokens
from eval.shared.judge import judge_llm
from eval.shared.questions import load_questions, save_results

# How many questions are evaluated at the same time. LM Studio handles 4
# requests in parallel, so more workers would only wait in its queue.
WORKERS = 4

HERE = Path(__file__).parent                        # eval/rag/
QUESTIONS_FILE = HERE / "questions.jsonl"
RESULTS_FILE = HERE / "results" / "results.json"
REPORT_FILE = HERE / "results" / "report.html"


async def judge(row: dict, metrics: dict) -> dict:
    """Score one answered question with the three RAGAS judge metrics.

    Args:
        row: A question dict that already has `answer` and `contexts`.
        metrics: The RAGAS metric objects created in `main()`.

    Returns:
        `{"context_recall", "faithfulness", "answer_accuracy"}`, each from 0 to 1.
    """
    contexts = [chunk["text"] for chunk in row["contexts"]]  # RAGAS wants only the text

    # The three judge calls don't depend on each other, so run them together.
    recall, faith, acc = await asyncio.gather(
        # Compare the reference answer with the chunks: was everything needed found?
        metrics["recall"].ascore(
            user_input=row["question"], retrieved_contexts=contexts, reference=row["reference"]),
        # Compare the answer with the chunks: is each claim backed by them?
        metrics["faith"].ascore(
            user_input=row["question"], response=row["answer"], retrieved_contexts=contexts),
        # Compare the answer with the reference: is it correct?
        metrics["acc"].ascore(
            user_input=row["question"], response=row["answer"], reference=row["reference"]),
    )
    return {
        "context_recall": recall.value,
        "faithfulness": faith.value,
        "answer_accuracy": acc.value,
    }


def log(t0: float, message: str) -> None:
    """Print a message with the seconds since the eval started, e.g. `[ 12.3s] ...`.

    The timestamps show the parallel work: several START lines at 0.0s, and a
    new START right after each DONE (a slot freed up, the next question takes it).
    """
    print(f"[{time.perf_counter() - t0:5.1f}s] {message}")


# How many questions are running right now. Only shown in the log, to make the
# parallel work visible. A plain global is safe: asyncio runs on one thread and
# switches tasks only at an `await`, never in the middle of `running += 1`.
running = 0


async def evaluate_one(num: int, row: dict, metrics: dict, limit: asyncio.Semaphore, t0: float,
                       tracer, answer_fn) -> dict:
    """Answer one question, check retrieval, run the judges, and log START and DONE.

    Args:
        num: Question number (1, 2, ...), only used in the log.
        row: One question dict from the question file.
        metrics: The RAGAS metric objects.
        limit: Semaphore that lets at most WORKERS questions run at once.
        t0: Time the eval started, for the log timestamps.
        tracer: From start_tracing(); groups this question's calls into one Phoenix trace.
        answer_fn: The system to test: plain RAG's answer(), or the LangGraph agent's.

    Returns:
        The same row with the answer, the contexts, the latency and all scores added.
    """
    global running
    async with limit:  # wait here until one of the WORKERS slots is free
        # One trace per question: the answer call and the judge calls all land under it.
        with tracer.start_as_current_span(f"Q{num}: {row['question']}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")   # how Phoenix labels this step
            span.set_attribute("input.value", row["question"])
            row["trace_id"] = current_trace_id()  # so the report can link to it
            running += 1
            log(t0, f"START Q{num:<2} (running: {running})  {row['question'][:55]}")

            # 1. Run the RAG pipeline and time it. answer() is normal (sync) code,
            #    so run it in a thread to avoid blocking the other questions.
            start = time.perf_counter()
            out = await asyncio.to_thread(answer_fn, row["question"])
            row["answer"] = out["answer"]
            row["contexts"] = out["contexts"]
            row["searches"] = out.get("searches", 1)          # the LangGraph agent may search twice
            row["query"] = out.get("query", row["question"])  # ...with a rewritten query
            row["latency_s"] = round(time.perf_counter() - start, 1)
            row["tokens"] = answer_tokens(row["question"], out)  # instructions / context / question / output
            span.set_attribute("output.value", row["answer"])

            # 2. Retrieval check without an LLM: is the expected PDF among the found
            #    chunks? Questions with source=None (no single right PDF) always count as a hit.
            found_sources = {chunk["source"] for chunk in out["contexts"]}
            row["retrieval_hit"] = float(row["source"] is None or row["source"] in found_sources)

            # 3. LLM-judge scores.
            row |= await judge(row, metrics)

            tokens = row["tokens"]
            running -= 1
            log(t0, f"DONE  Q{num:<2} (running: {running})  hit={row['retrieval_hit']:.0f} "
                    f"recall={row['context_recall']:.2f} faith={row['faithfulness']:.2f} "
                    f"acc={row['answer_accuracy']:.2f}  answer took {row['latency_s']}s")
            log(t0, f"      Q{num:<2} tokens: answer {tokens['input']:,} in (context {tokens['context']:,}) "
                    f"+ {tokens['output']:,} out (reasoning {tokens['reasoning']:,})")
            return row


async def main(answer_fn=answer, results_dir=RESULTS_FILE.parent, title="RAG", project="rag-eval"):
    """Evaluate all questions, print the averages, and save the results and the report.

    The arguments let Doc 4 reuse this same eval for the LangGraph agent.

    Args:
        answer_fn: The system to test. Takes a question, returns `{"answer", "contexts", "usage"}`.
        results_dir: Folder for results.json and report.html.
        title: Name of the system, shown in the HTML report.
        project: The Phoenix project the traces go to.
    """
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow non-ASCII characters
    tracer = start_tracing(project, judge=True)  # record the system and the judge in Phoenix

    # One judge shared by the three metrics.
    llm = judge_llm()
    metrics = {
        "recall": ContextRecall(llm=llm),
        "faith": Faithfulness(llm=llm),
        "acc": AnswerAccuracy(llm=llm),
    }
    rows = load_questions(QUESTIONS_FILE)

    # Start every question at once. The semaphore keeps only WORKERS running,
    # and gather() returns the results in the original question order.
    print(f"{len(rows)} questions, {WORKERS} at a time\n")
    t0 = time.perf_counter()
    limit = asyncio.Semaphore(WORKERS)
    results = await asyncio.gather(
        *(evaluate_one(num, row, metrics, limit, t0, tracer, answer_fn) for num, row in enumerate(rows, start=1)))
    print(f"\ntotal time: {time.perf_counter() - t0:.0f}s with {WORKERS} workers")

    # The average of each score is the number that goes in the README results table.
    keys = ["retrieval_hit", "context_recall", "faithfulness", "answer_accuracy", "latency_s", "searches"]
    means = [f"{key}={sum(r[key] for r in results) / len(results):.2f}" for key in keys]
    print("\nMEAN  " + "  ".join(means))

    # Answer-call tokens over all questions. The judge's tokens, and the total of every
    # call, are in Phoenix: project "rag-eval" shows them at the top of its page.
    def total(key):
        return sum(r["tokens"][key] for r in results)
    print(f"TOKENS  answer calls: {total('input'):,} in + {total('output'):,} out"
          f"  |  judge + everything: see Phoenix, project {project}")

    # Save every row (answer, chunks, scores, tokens) so you can read the failing ones later,
    # and the same data as an HTML page for the browser.
    results_dir = Path(results_dir)
    save_results(results, results_dir / "results.json")
    write_rag_report(results, results_dir / "report.html", WORKERS, title, project)
    print(f"\nreport: {(results_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
