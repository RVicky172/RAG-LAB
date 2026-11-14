import numpy as np
import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")  # key is ignored locally



a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.5])
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(round(cos, 4))   # close to 1.0 → pointing the same direction


emb = client.embeddings.create(model=os.getenv("EMBED_MODEL"),
                               input=["search_query: reset my password",
                                      "search_document: How to recover your login credentials"])
v = [np.array(d.embedding) for d in emb.data]
print(len(v[0]), float(v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))))
