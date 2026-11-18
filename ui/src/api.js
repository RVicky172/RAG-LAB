/**
 * The only file that talks to the FastAPI server (app/api.py).
 *
 * Why keep it separate: the components then only deal with data, and if an
 * endpoint changes there is one place to fix.
 *
 * All URLs start with /api: in development Vite forwards them to
 * localhost:8000 (vite.config.js); in production FastAPI serves this page
 * itself, so they go to the same server.
 */

/**
 * Ask the server whether it is up, which LLM it uses and how many chunks Qdrant holds.
 *
 * @returns {Promise<{status: string, llm: string, llm_host: string, chunks: number}>}
 */
export async function getHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Ask one system one question.
 *
 * @param {string} system - "rag", "agent", "multi-agent" or "langgraph".
 * @param {string} question - The user's question.
 * @returns {Promise<{system: string, answer: string, sources: string[],
 *   steps: {name: string, input: string}[], seconds: number, trace_id: string}>}
 * @throws {Error} With the server's reason, e.g. when LM Studio isn't running.
 */
export async function ask(system, question) {
  const res = await fetch(`/api/ask/${system}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    // FastAPI puts the reason in "detail". It's a string for our own errors (502)
    // and a list for validation errors (422), so fall back to the status code.
    const body = await res.json().catch(() => ({}));
    const reason = typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
    throw new Error(reason);
  }
  return res.json();
}
