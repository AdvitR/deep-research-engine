from langchain_openai import ChatOpenAI
import os


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )
