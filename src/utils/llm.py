import os

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )


model = init_chat_model("gpt-5-mini", temperature=0)
