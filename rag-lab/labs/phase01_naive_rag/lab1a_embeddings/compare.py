import os, numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
load_dotenv()

sents = ["How do I reset my password?", "I forgot my login credentials",
         "The quarterly revenue grew 12%", "Error code HX-4410 on startup"]

small = SentenceTransformer("all-MiniLM-L6-v2")              # 384 dims, 256-token limit, tiny
E1 = small.encode(sents, normalize_embeddings=True)

lms = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")
r = lms.embeddings.create(model=os.getenv("EMBED_MODEL"),     # nomic-embed-text-v1.5
                          input=["search_document: " + s for s in sents])
E2 = np.array([d.embedding for d in r.data]); E2 /= np.linalg.norm(E2, axis=1, keepdims=True)

print(np.round(E1 @ E1.T, 2)); print(np.round(E2 @ E2.T, 2))
