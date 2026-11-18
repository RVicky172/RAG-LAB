"""
Settings for the whole project, kept in one place.

Every value is read from an environment variable. Locally these come from the
`.env` file. On AWS they come from the server's environment. If a variable is
not set, the default below is used, and the defaults match the local setup
(LM Studio + Qdrant in Docker).

Why do it this way? The code never changes between local and AWS. Only the
settings change.
"""

import os

from dotenv import load_dotenv

# Copy the values from `.env` into os.environ. Variables that are already set
# (for example on AWS) are NOT overwritten.
load_dotenv()

# --- Vector database (Qdrant) -------------------------------------------------
# Where Qdrant runs, and the name of the "table" (collection) holding our chunks.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "papers")

# --- Embedding model ----------------------------------------------------------
# Turns text into a vector (a list of numbers) so similar texts end up close
# together. Small, fast, and runs on the CPU through fastembed.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chat model (LLM) ---------------------------------------------------------
# LM Studio (local) and Bedrock (AWS) both speak the OpenAI API, so we only
# need a model name, a URL and a key.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")  # LM Studio accepts any key

# The most tokens one LLM call can hold (prompt + reply together). Locally it
# is what we loaded the model with (`lms load ... --context-length 32768`).
# On Bedrock gpt-oss-20b allows 128k, so set CONTEXT_WINDOW=131072 there.
# Only used to report how full the window is.
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "32768"))

# --- Tracing (Arize Phoenix) -----------------------------------------------------
# Phoenix runs locally in Docker (docker-compose.yml) and records every LLM call,
# tool call and agent step, with tokens and timing. Open PHOENIX_URL to see them.
# Set TRACING=false to run without it (e.g. when Phoenix isn't running).
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
TRACING = os.getenv("TRACING", "true").lower() == "true"
