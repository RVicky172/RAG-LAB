# rag/llm.py
import os, re
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio")          # lmstudio | bedrock
GEN_MODEL = os.getenv("GEN_MODEL")
HELPER_MODEL = os.getenv("HELPER_MODEL", GEN_MODEL)
_THINK = re.compile(r"<think>.*?</think>", re.S)           # strip reasoning blocks if a model emits them

if PROVIDER == "lmstudio":
    from openai import OpenAI
    _client = OpenAI(base_url=os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1"), api_key="lm-studio")
else:
    import boto3
    _client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"))

def complete(prompt: str, system: str = "", model: str | None = None,
             max_tokens: int = 1500, temperature: float | None = None) -> str:
    model = model or GEN_MODEL
    if PROVIDER == "lmstudio":
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = _client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens,
                                            temperature=0.2 if temperature is None else temperature)
        text = r.choices[0].message.content or ""
    else:  # Amazon Bedrock Converse API (Phase 09)
        kwargs = dict(modelId=model,
                      messages=[{"role": "user", "content": [{"text": prompt}]}],
                      inferenceConfig={"maxTokens": max_tokens})
        if system:
            kwargs["system"] = [{"text": system}]
        if temperature is not None:        # some newer Bedrock models reject temperature; send only if set
            kwargs["inferenceConfig"]["temperature"] = temperature
        r = _client.converse(**kwargs)
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    return _THINK.sub("", text).strip()

def helper(prompt: str, **kw) -> str:
    """Cheap, fast model for rewriting, grading, contextualizing (Qwen3-4B locally)."""
    return complete(prompt, model=HELPER_MODEL, temperature=0, **kw)
