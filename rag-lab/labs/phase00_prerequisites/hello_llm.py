# hello_llm.py
import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.getenv("LMSTUDIO_URL"), api_key="lm-studio")  # key is ignored locally

t0 = time.perf_counter()
resp = client.chat.completions.create(
    model=os.getenv("GEN_MODEL"),
    messages=[{"role": "system", "content": "You are a concise assistant."},
              {"role": "user", "content": "Explain RAG in two sentences."}],
    temperature=0.2, max_tokens=300,stream=True,stream_options={"include_usage": True},
)
dt = time.perf_counter() - t0

parts, usage, ttft = [], None, None
for chunk in resp:
    if chunk.usage:                      # the last chunk has usage and an empty choices list
        usage = chunk.usage
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta.content
    if delta:                            # the first and last chunks can be None
        if ttft is None:
            ttft = time.perf_counter() - t0
        print(delta, end="", flush=True) # print text as it arrives
        parts.append(delta)

dt = time.perf_counter() - t0
full_text = "".join(parts)
print()

if usage:
    print(f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
          f"ttft={ttft:.2f}s total={dt:.1f}s ~{usage.completion_tokens/dt:.0f} tok/s")
# print(resp.choices[0].message.content)
# u = resp.usage
# print(f"prompt={u.prompt_tokens} completion={u.completion_tokens} "
#       f"time={dt:.1f}s  ~{u.completion_tokens/dt:.0f} tok/s")
