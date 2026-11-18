/**
 * One system's answer to one question: the answer text, the steps it took,
 * its sources, the time, and a link to the full trace in Phoenix.
 *
 * Showing the steps next to the answer is the point of this UI: you can see
 * that RAG always searches once, that the agent sometimes skips the search,
 * and that the multi-agent supervisor delegates instead of searching.
 */

import { PHOENIX_URL, SYSTEMS } from "./systems.js";

// When tracing is off (TRACING=false) the API returns an all-zero trace id.
const NO_TRACE = "0".repeat(32);

/**
 * Draw one answer card.
 *
 * @param {object} props
 * @param {string} props.systemId - Which system answered, e.g. "rag".
 * @param {{status: "loading"|"done"|"error", data?: object, error?: string}} props.result
 * @returns {JSX.Element}
 */
export default function AnswerCard({ systemId, result }) {
  const system = SYSTEMS.find((s) => s.id === systemId);

  return (
    <article className="card">
      <header className="card-head">
        <span className="card-title">{system.name}</span>
        <span className="muted">{system.doc}</span>
        {result.status === "done" && <span className="time">{result.data.seconds} s</span>}
      </header>

      {result.status === "loading" && <p className="muted thinking">Thinking…</p>}

      {result.status === "error" && (
        <p className="error">
          {result.error}
          <br />
          <span className="muted">Is LM Studio running with the model loaded, and Qdrant up?</span>
        </p>
      )}

      {result.status === "done" && <Details data={result.data} />}
    </article>
  );
}

/**
 * Show the model's **bold** markdown as bold text.
 *
 * gpt-oss writes a little markdown, mostly **bold**. A full markdown library
 * would be one more dependency, so we handle only this one case: splitting
 * on "**" makes every odd-numbered piece the bold part.
 *
 * @param {string} text - The answer text.
 * @returns {Array<string|JSX.Element>} Plain strings and <strong> elements, in order.
 */
function withBold(text) {
  return text.split("**").map((piece, i) => (i % 2 === 1 ? <strong key={i}>{piece}</strong> : piece));
}

/**
 * The body of a finished card.
 *
 * @param {object} props
 * @param {object} props.data - The JSON the API returned (see app/api.py, class Answer).
 * @returns {JSX.Element}
 */
function Details({ data }) {
  const hasTrace = data.trace_id !== NO_TRACE;

  return (
    <>
      {/* pre-wrap keeps the model's line breaks; the answer is plain text, not HTML. */}
      <p className="answer">{withBold(data.answer)}</p>

      <h4>Steps ({data.steps.length})</h4>
      {data.steps.length === 0 ? (
        <p className="muted">No tools used: answered directly.</p>
      ) : (
        <ol className="steps">
          {data.steps.map((step, i) => (
            <li key={i}>
              <code>{step.name}</code> {step.input}
            </li>
          ))}
        </ol>
      )}

      {data.sources.length > 0 && (
        <>
          <h4>Sources</h4>
          <div className="chips">
            {data.sources.map((source) => (
              <span key={source} className="chip">{source}</span>
            ))}
          </div>
        </>
      )}

      {hasTrace && (
        <a className="trace" href={`${PHOENIX_URL}/redirects/traces/${data.trace_id}`} target="_blank" rel="noreferrer">
          Open trace in Phoenix (tokens, timing, every LLM call) →
        </a>
      )}
    </>
  );
}
