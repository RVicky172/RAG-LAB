/**
 * The four systems the UI can talk to, with a one-line reminder of how each works.
 *
 * The `id` is the {system} part of the URL POST /api/ask/{system} in app/api.py.
 */

export const SYSTEMS = [
  { id: "rag", name: "RAG", doc: "Doc 1", how: "Always searches once, then answers from the chunks." },
  { id: "agent", name: "Agent", doc: "Doc 2", how: "One LLM that decides when to search or calculate." },
  { id: "multi-agent", name: "Multi-agent", doc: "Doc 3", how: "A supervisor that delegates to a researcher and an analyst." },
  { id: "langgraph", name: "LangGraph", doc: "Doc 4", how: "Searches, grades the chunks, rewrites the query once if needed." },
];

// Questions to click, one per kind of task: a fact, arithmetic, a comparison, small talk.
export const EXAMPLES = [
  "How many attention heads does the base Transformer use?",
  "How much higher is the big Transformer EN-DE BLEU than the base model?",
  "Which two tasks does the Lost in the Middle paper use?",
  "What does RAGAS measure with faithfulness?",
  "Hi! What can you help me with?",
];

// Where Phoenix shows a trace. Docker publishes it on this port, so it also works with the container.
export const PHOENIX_URL = "http://localhost:6006";
