from state.research_state import ResearchState
from langchain_core.messages import HumanMessage
from utils.llm import model


def clarity_scorer(state: ResearchState):
    print("=== Clarity Scorer Agent ===")
    prompt = f"""
You are evaluating the clarity of a user-submitted research query.

Clarity is defined as how **specific, interpretable, and actionable** the query is for an AI research assistant. A clear query contains enough context, scope, and intent to be understood and processed without needing follow-up clarification.

### Instructions:
- Return **only a float between 0.0 and 1.0**
- **0.0** = completely vague or ambiguous
- **1.0** = perfectly clear and immediately actionable
- No explanation, no formatting, no justification — only the number

### Scoring Guide with Examples:

- **1.0** → Fully clear, precise, scoped
    > "What are the top 3 publicly reported causes of turbine blade failures in GE CF6-80 engines between 2015 and 2023?"

- **0.8** → Mostly clear, some minor scope ambiguity
    > "How does long-term exposure to PFAS affect human fertility?"

- **0.6** → Somewhat clear, but missing key context or constraints
    > "What are the effects of pollution on public health?"

- **0.4** → Vague or broad; lacks specificity or target
    > "Tell me about AI in medicine."

- **0.2** → Highly ambiguous, could mean many things
    > "What's going on with climate?"

- **0.0** → Nonspecific, contextless, or meaningless
    > "Can you look into this for me?"

### Query to Evaluate:
{state['user_query']}
"""

    response = model.invoke([HumanMessage(content=prompt)]).content
    score = float(response.strip())

    print("=== Clarity Scorer Result ===")
    print({"clarity_score": score, "clarification_needed": score < 0.6})
    return {"clarity_score": score, "clarification_needed": score < 0.6}
