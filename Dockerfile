# The app image: FastAPI + the four systems + the built React UI (Doc 5).
# Build and run: docker compose --profile app up -d --build

# --- Stage 1: build the React UI --------------------------------------------------
# Node is only needed to BUILD the UI (ui/ -> ui/dist, plain HTML/JS/CSS).
# The final image doesn't need Node at all, so it is a separate stage.
FROM node:24-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui ./
RUN npm run build

# --- Stage 2: the Python app ------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /app

# Install the libraries first: this layer is cached until uv.lock changes,
# so rebuilding after a code change takes seconds.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# Download the embedding model at build time, so the container starts fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY eval ./eval
COPY data/raw ./data/raw
COPY --from=ui /ui/dist ./ui/dist

CMD ["uvicorn", "app.api:api", "--host", "0.0.0.0", "--port", "8000"]
