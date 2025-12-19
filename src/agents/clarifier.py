from utils.llm import get_llm
from state.research_state import ResearchState
from langchain_core.messages import HumanMessage
from utils.llm import model


def clarifier(state: ResearchState) -> dict:
    prompt = f"""
    You are assisting with a research task. The user's original query is:

    User query:
    "{state['user_query']}"

    The system has determined that some clarification in the query is needed.
    Identify the single most critical ambiguity that would materially affect
    the research outcome, and ask ONE concise clarification question.

    Do NOT ask about minor details, formatting preferences, or optional scope
    extensions. If the query is already sufficiently actionable, return the
    string "NO_CLARIFICATION_NEEDED".

    Your question (or "NO_CLARIFICATION_NEEDED"):
    """

    question = model.invoke([HumanMessage(content=prompt)]).content
    # TODO: assuming user responds externally for now
    clarified = state["user_query"] + " (clarified)"

    return {"clarified_query": clarified}
