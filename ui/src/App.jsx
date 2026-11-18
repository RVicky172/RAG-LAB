/**
 * The whole page: pick a system (or compare all four), ask a question, see the answers.
 *
 * Every question becomes a "turn": the question plus one result per system
 * asked. Each system's answer fills in as soon as it arrives, so in "Compare
 * all" mode you also see which system is fastest.
 *
 * Each question is answered on its own: the systems have no chat memory
 * (Docs 1-4 built them that way), so a follow-up like "and the big one?" won't work.
 */

import { useEffect, useState } from "react";

import AnswerCard from "./AnswerCard.jsx";
import { ask, getHealth } from "./api.js";
import { EXAMPLES, SYSTEMS } from "./systems.js";

const COMPARE = "compare";

/**
 * The page component.
 *
 * @returns {JSX.Element}
 */
export default function App() {
  const [health, setHealth] = useState(null); // null = still checking, "down" = no answer
  const [mode, setMode] = useState("rag");    // a system id, or COMPARE for all four
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]);     // newest first

  // Check the server once when the page opens.
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth("down"));
  }, []);

  // Don't send a new question while one is still being answered: locally there
  // is one GPU, and parallel questions would only slow each other down.
  const busy = turns.some((turn) => Object.values(turn.results).some((r) => r.status === "loading"));

  /**
   * Store one system's result in one turn. Uses the "previous state" form of
   * setTurns because several answers can arrive at about the same time.
   *
   * @param {number} turnId - Which turn.
   * @param {string} systemId - Which system's result.
   * @param {object} result - `{status, data}` or `{status, error}`.
   */
  function setResult(turnId, systemId, result) {
    setTurns((prev) =>
      prev.map((turn) =>
        turn.id === turnId ? { ...turn, results: { ...turn.results, [systemId]: result } } : turn,
      ),
    );
  }

  /**
   * Send the question to the chosen system(s) and add a new turn for it.
   *
   * @param {string} text - The question.
   */
  function submit(text) {
    const q = text.trim();
    if (!q || busy) {
      return;
    }

    const systemIds = mode === COMPARE ? SYSTEMS.map((s) => s.id) : [mode];
    const turnId = Date.now();

    // Start every result as "loading", then fill each one in when its answer arrives.
    const results = Object.fromEntries(systemIds.map((id) => [id, { status: "loading" }]));
    setTurns((prev) => [{ id: turnId, question: q, systemIds, results }, ...prev]);
    setQuestion("");

    for (const id of systemIds) {
      ask(id, q)
        .then((data) => setResult(turnId, id, { status: "done", data }))
        .catch((err) => setResult(turnId, id, { status: "error", error: err.message }));
    }
  }

  /**
   * Enter sends, Shift+Enter makes a new line (like most chat apps).
   *
   * @param {KeyboardEvent} e
   */
  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(question);
    }
  }

  const selected = SYSTEMS.find((s) => s.id === mode);

  return (
    <div className="page">
      <header className="top">
        <h1>Paper assistant</h1>
        <Status health={health} />
      </header>

      <section className="panel">
        <div className="modes" role="tablist">
          {SYSTEMS.map((s) => (
            <button key={s.id} role="tab" aria-selected={mode === s.id} onClick={() => setMode(s.id)}>
              {s.name}
            </button>
          ))}
          <button role="tab" aria-selected={mode === COMPARE} onClick={() => setMode(COMPARE)}>
            Compare all
          </button>
        </div>
        <p className="muted how">
          {selected ? selected.how : "Sends the question to all four systems and shows the answers side by side."}
        </p>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(question);
          }}
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about the Transformer, RAG, Lost in the Middle or RAGAS papers…"
            rows={2}
          />
          <button type="submit" className="primary" disabled={busy || !question.trim()}>
            {busy ? "Answering…" : "Ask"}
          </button>
        </form>

        <div className="chips examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => submit(ex)} disabled={busy}>
              {ex}
            </button>
          ))}
        </div>
      </section>

      {turns.map((turn) => (
        <section key={turn.id} className="turn">
          <p className="question">{turn.question}</p>
          <div className={turn.systemIds.length > 1 ? "grid" : ""}>
            {turn.systemIds.map((id) => (
              <AnswerCard key={id} systemId={id} result={turn.results[id]} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

/**
 * The status line in the header: which LLM answers and how many chunks Qdrant holds.
 *
 * @param {object} props
 * @param {object|string|null} props.health - The /api/health JSON, "down", or null while checking.
 * @returns {JSX.Element}
 */
function Status({ health }) {
  if (health === null) {
    return <span className="status muted">checking…</span>;
  }
  if (health === "down") {
    return <span className="status bad">API not reachable: is uvicorn running on port 8000?</span>;
  }
  if (health.chunks === 0) {
    return <span className="status bad">Qdrant is empty: run the ingest</span>;
  }
  return (
    <span className="status ok">
      {health.llm} on {health.llm_host} · {health.chunks} chunks
    </span>
  );
}
