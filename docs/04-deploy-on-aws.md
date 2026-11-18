# Doc 4: Deploy on AWS

**Time:** ~2 hours  **You build:** the same three systems (RAG, agent, multi-agent) running on AWS behind an HTTP API, with the LLM served by Amazon Bedrock and the same evals run there.

---

## 1. What changes and what doesn't

```
LOCAL (Docs 1-3)                         AWS (this doc)
──────────────────                       ──────────────────────────────────────────
your Python  ──► Qdrant (Docker)         EC2 instance (one small server)
     │                                    ┌──────────── docker compose ────────────┐
     └──────► LM Studio (your GPU)        │ app (FastAPI) ──► qdrant (vector DB)   │
                                          └───┬────────────────────────────────────┘
                                              │ HTTPS, Bedrock API key
                                              ▼
                                          Amazon Bedrock: gpt-oss-20b (same model!)
```

- **Code: no changes.** LM Studio and Bedrock both speak the OpenAI chat API, so only 3 lines of `.env` change.
- **LLM → Amazon Bedrock.** AWS runs the model; you pay per token, with no GPU to manage. It's the same `gpt-oss-20b`, so local and AWS eval numbers are comparable.
- **App + Qdrant → one EC2 server running Docker Compose.** It's the simplest deployment you can fully see and understand. Section 8 lists what larger companies use instead.

**Before you start:** an AWS account, a budget alert (Billing → Budgets → e.g. $10), and your code pushed to GitHub. This doc uses region **us-west-2 (Oregon)**. Check the [gpt-oss-20b model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-20b.html) for the regions it's in today.

---

## 2. Bedrock: get an API key and test it (20 min)

1. AWS Console → **Amazon Bedrock** (region us-west-2) → **API keys** → generate a **long-term API key** with a short expiry (e.g. 1 day; this is for learning). Copy it.
2. Test it from your laptop (Git Bash):

```bash
export BEDROCK_KEY="paste-key-here"
curl -s https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $BEDROCK_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openai.gpt-oss-20b-1:0", "messages": [{"role": "user", "content": "Say hi in 3 words"}]}'
```

3. **Point your local code at Bedrock.** This is the key moment: no code changes. Edit `.env`:

```bash
LLM_MODEL=openai.gpt-oss-20b-1:0
LLM_BASE_URL=https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1
LLM_API_KEY=paste-key-here
```

```bash
uv run python -m app.rag          # same answer, now generated in AWS
lms unload --all                  # you no longer need the local GPU model
```

> The API key is a secret. `.env` is in `.gitignore`. Never commit it, and never paste it into code.

---

## 3. Put the app behind an HTTP API (15 min)

Real services are called over HTTP, not from a terminal. **FastAPI** turns your three functions into web endpoints.

**`app/api.py`**
```python
"""HTTP API: the same three systems, behind one web service."""
from fastapi import FastAPI
from pydantic import BaseModel
from app import agent, multi_agent, rag

api = FastAPI(title="Paper assistant")

class Question(BaseModel):
    question: str

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/rag")
def ask_rag(q: Question):
    out = rag.answer(q.question)
    return {"answer": out["answer"], "sources": [f"{c['source']} p.{c['page']}" for c in out["contexts"]]}

@api.post("/agent")
def ask_agent(q: Question):
    return {"answer": agent.run(q.question)["answer"]}

@api.post("/multi-agent")
def ask_multi_agent(q: Question):
    return {"answer": multi_agent.run(q.question)["answer"]}
```

```bash
uv run uvicorn app.api:api --port 8000
# open http://localhost:8000/docs  -> FastAPI's built-in page to try each endpoint
```

---

## 4. Package it as a container (20 min)

A container image = your code + Python + every library, frozen. It runs the same on your laptop and on AWS.

**`Dockerfile`**
```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY eval ./eval
COPY data/raw ./data/raw
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# Download the embedding model at build time so the container starts fast
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
CMD ["uvicorn", "app.api:api", "--host", "0.0.0.0", "--port", "8000"]
```

**`.dockerignore`** (keeps secrets and junk out of the image)
```
.venv
.env
.git
data/qdrant
eval/results
```

**Add the app to `docker-compose.yml`** (below the `qdrant` service):
```yaml
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      QDRANT_URL: http://qdrant:6333   # inside Compose, services find each other by name
    depends_on:
      - qdrant
```

Test the whole stack locally (with the Bedrock `.env` from step 2):
```bash
docker compose up -d --build                       # first build ~2 min, image ~1.1 GB
docker compose exec app python -m app.ingest       # fill Qdrant from inside the container
curl -s -X POST localhost:8000/rag -H "Content-Type: application/json" \
  -d '{"question": "Which two tasks does the Lost in the Middle paper use?"}'
```

If this works on your laptop, it will work on EC2. That's the point of containers.

---

## 5. Launch the server on EC2 (30 min)

1. Commit and push (the PDFs in `data/raw/` are committed; `.env` is not).
2. Console → **EC2** → **Launch instance**:
   - Name `paper-assistant`, image **Amazon Linux 2023**, type **t3.medium** (2 vCPU, 4 GB RAM: enough, since the LLM runs in Bedrock).
   - Key pair: create one, download the `.pem`.
   - Network → security group: allow **SSH (22)** and **Custom TCP 8000**, both from **My IP** only.
   - Storage: 20 GB.
3. Connect and install Docker:

```bash
ssh -i paper-assistant.pem ec2-user@<public-ip>

sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user && newgrp docker
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version
```

4. Get the code, add secrets, start:

```bash
git clone https://github.com/<you>/<repo>.git app && cd app
cp .env.example .env && nano .env          # set the 3 Bedrock lines from step 2
docker compose up -d --build
docker compose exec app python -m app.ingest
```

5. From **your laptop**:

```bash
curl -s http://<public-ip>:8000/health
curl -s -X POST http://<public-ip>:8000/multi-agent -H "Content-Type: application/json" \
  -d '{"question": "How much higher is the big Transformer EN-DE BLEU than the base model?"}'
```

Or open `http://<public-ip>:8000/docs` in your browser.

---

## 6. Evaluate in AWS (15 min)

A deployment isn't done until it's measured. Run the same three evals on the server:

```bash
docker compose exec app python -m eval.rag_eval
docker compose exec app python -m eval.agent_eval
docker compose exec app python -m eval.multi_agent_eval
```

Add an "AWS" row for each to the README table and compare with local:
- **Quality** should be close (same model, same data). A big drop means something differs: check the ingest chunk count and the `.env`.
- **Latency** changes: network round-trips to Bedrock versus your local GPU.
- **Cost:** Bedrock console → **Usage**, or CloudWatch metrics for token counts. Multiply by the price on the Bedrock pricing page to get cost per question. Multi-agent uses more tokens per question, so check by how much.

**If the judge fails with a `response_format` error on Bedrock:** in `judge_llm()` change `mode=instructor.Mode.JSON_SCHEMA` to `mode=instructor.Mode.TOOLS` (structured output through tool calling) and drop `reasoning_effort` if Bedrock rejects it.

---

## 7. Tear down (5 min): don't skip this

```bash
docker compose down            # on the server
```
- EC2 → select instance → **Terminate** (stopping still bills for the disk).
- Bedrock → API keys → **delete** the key.
- Check Billing the next day.

---

## 8. How bigger companies do the same thing

You built the simple version on purpose. Each row below replaces one piece; the app code stays the same.

| You did | Production teams usually use | Why |
|---|---|---|
| One EC2 + Docker Compose | **ECS Fargate** or EKS, image in **ECR** | auto-restart, scaling, no server to patch |
| Qdrant container on the same box | **Qdrant Cloud**, **OpenSearch Serverless**, or **RDS Postgres + pgvector** | managed backups, high availability |
| API key in `.env` | **IAM role** on the task + **Secrets Manager** | no long-lived keys on disk |
| Port 8000 open to your IP | **Load balancer + HTTPS + auth** (Cognito / API keys) | security |
| `print` + eval JSON | **CloudWatch** logs, tracing (LangSmith / Langfuse) | debug real user traffic |
| Run evals by hand | **Evals in CI** as a gate before each deploy | no quality regressions |

---

## Checklist

- [ ] I switched the LLM from local to Bedrock by changing only `.env`.
- [ ] I can explain what a container image is and why "works locally in Docker" means "works on EC2".
- [ ] I called all three endpoints on the EC2 server.
- [ ] I ran the evals in AWS and compared quality, latency and cost with local.
- [ ] I terminated the instance and deleted the API key.

**Done.** Back to the [README](../README.md) to fill in your final results table.
