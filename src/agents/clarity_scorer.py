from utils.llm import get_llm
from state.research_state import ResearchState


def clarity_scorer(state: ResearchState):
    llm = get_llm()

    prompt = f"""
    The following query was provided by a user to be processed by an AI
    research agent. Rate how clear and well-specified the user query is.

    Query:
    {state['user_query']}

    Return ONLY a float between 0 (completely unclear) and 1 (perfectly clear).
    Nothing else.
    """

    # TODO: might need to be modified per OpenAI API changes
    response = llm.predict(prompt)
    score = float(response.content.strip())

    return {
        "clarity_score": score,
        "clarification_needed": score < 0.6
    }
