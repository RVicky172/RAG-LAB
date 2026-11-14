# rag/embed.py
import os, json
import numpy as np
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("EMBED_PROVIDER", "lmstudio")        # lmstudio | st | bedrock
MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

# Query/document prefixes per model family (read each model card!)
PREFIXES = {
    "nomic":  ("search_query: ", "search_document: "),
    # Exactly the model card's retrieval instruction. Our own wording ("Given a question,
    # retrieve passages that answer it\nQuery: ") cost 15 points of hit@5 (Phase 02, Step 6).
    "qwen3":  ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", ""),
    "bge-small": ("Represent this sentence for searching relevant passages: ", ""),
}
QP, DP = next((v for k, v in PREFIXES.items() if k in MODEL.lower()), ("", ""))

if PROVIDER == "lmstudio":
    from openai import OpenAI
    _client = OpenAI(base_url=os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1"), api_key="lm-studio")
elif PROVIDER == "st":                                      # sentence-transformers on your GPU
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer(MODEL, device="cuda")
else:                                                       # Bedrock Titan V2 (Phase 09)
    import boto3
    _client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"))

def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)

def _embed(texts: list[str], batch: int = 64) -> np.ndarray:
    if PROVIDER == "st":
        return _st.encode(texts, normalize_embeddings=True, batch_size=32)
    out = []
    if PROVIDER == "lmstudio":
        for i in range(0, len(texts), batch):
            r = _client.embeddings.create(model=MODEL, input=texts[i:i + batch])
            out += [d.embedding for d in r.data]
    else:
        for t in texts:                                     # Titan embeds one text per call
            body = json.dumps({"inputText": t, "dimensions": 1024, "normalize": True})
            r = _client.invoke_model(modelId=MODEL, body=body)
            out.append(json.loads(r["body"].read())["embedding"])
    return _normalize(np.array(out, dtype="float32"))

def embed_docs(texts: list[str]) -> np.ndarray:
    return _embed([DP + t for t in texts])

def embed_query(q: str) -> np.ndarray:
    return _embed([QP + q])[0]
