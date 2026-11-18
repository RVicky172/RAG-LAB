"""
Creates the chat model (LLM) that the rest of the project uses.

LM Studio (local) and Amazon Bedrock (AWS) both speak the OpenAI chat API, so
one function covers both. Moving to AWS only changes the values in `.env`.
"""

from langchain.chat_models import init_chat_model

from app.shared import config


def get_llm():
    """Return a LangChain chat model connected to our LLM server.

    Returns:
        A LangChain chat model. Call `.invoke(prompt)` on it and read the
        reply text from `.content`.
    """
    return init_chat_model(
        config.LLM_MODEL,
        model_provider="openai",       # use the OpenAI client (LM Studio and Bedrock both support it)
        base_url=config.LLM_BASE_URL,  # where the server runs: LM Studio or Bedrock
        api_key=config.LLM_API_KEY,
        temperature=0,                 # no randomness, so eval runs are easier to compare
    )
