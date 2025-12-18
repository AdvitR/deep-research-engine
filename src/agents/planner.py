import json
from typing import List
from utils.llm import get_llm
from state.research_state import ResearchState, PlanStep

ALLOWED_METHODS = {"search", "analysis"}
ALLOWED_RISKS = {"low", "medium", "high"}


def _validate_plan(plan: List[dict]) -> List[PlanStep]:
    """
    Strictly validates planner output to ensure downstream safety.
    Raises ValueError on any violation.
    """
    if not isinstance(plan, list):
        raise ValueError("Planner output must be a list")

    validated: List[PlanStep] = []

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} is not a dict")

        # checking for all required keys
        required_keys = {"id", "goal", "method", "risk"}
        if set(step.keys()) != required_keys:
            raise ValueError(
                f"Step {i} keys must be {required_keys}, got {step.keys()}"
            )

        # asserting types and values for all required keys
        if not isinstance(step["id"], str):
            raise ValueError(f"Step {i} id must be a string")

        if not isinstance(step["goal"], str) or not step["goal"].strip():
            raise ValueError(f"Step {i} goal must be a non-empty string")

        if step["method"] not in ALLOWED_METHODS:
            raise ValueError(
                f"Step {i} method must be one of {ALLOWED_METHODS}"
            )

        if step["risk"] not in ALLOWED_RISKS:
            raise ValueError(
                f"Step {i} risk must be one of {ALLOWED_RISKS}"
            )

        validated.append(step)

    if len(validated) == 0:
        raise ValueError("Planner returned an empty plan")

    return validated


def planner(state: ResearchState) -> dict:
    llm = get_llm()

    query = state.get("clarified_query") or state["user_query"]

    prompt = f"""
    You are a research planning agent in a multi-agent system.

    Your job is to convert a high-level research query into a SMALL, EXECUTABLE
    sequence of research steps that can be carried out using external
    information sources (e.g., web search, public reports, datasets).

    Research Query:
    {query}

    Planning Objectives:
    - Decompose the query into concrete, factual sub-questions.
    - Each step should aim to retrieve or analyze real-world information.
    - Prefer steps that can be answered using public, authoritative sources
    (e.g., government reports, OECD, WHO, World Bank, peer-reviewed surveys).
    - Avoid steps that rely on speculation, private data, or judgment.

    Step Design Rules:
    For EACH step:
    1. The step must be independently executable.
    2. The step must correspond to ONE clear information need.
    3. The step must be feasible via search or simple analysis.
    4. If data availability is uncertain, mark the step as higher risk.

    Risk Levels:
    - "low": Data is very likely to exist in public sources.
    - "medium": Data likely exists but may require synthesis or proxies.
    - "high": Data may be incomplete, outdated, or unavailable.

    OUTPUT FORMAT (STRICT)
    Return ONLY a JSON list (no markdown, no explanation).

    Each list element must have EXACTLY these fields:
    - "id": short string identifier (e.g., "s1", "s2")
    - "goal": a precise description of what the step aims to find or compute
    - "method": one of ["search", "analysis"]
    - "risk": one of ["low", "medium", "high"]

    IMPORTANT CONSTRAINTS:
    - Do NOT answer the research question.
    - Do NOT include conclusions.
    - Do NOT include explanatory text.
    - Focus ONLY on planning executable steps.
    - Use the minimum number of steps required to answer the query well.

    Begin.
    """.strip()

    # TODO: fix llm invocation function
    raw_output = llm.invoke(prompt).content.strip()

    try:
        parsed = json.loads(raw_output)
        validated_plan = _validate_plan(parsed)
    except Exception as e:
        raise RuntimeError(
            f"Planner failed to produce valid output.\n"
            f"Raw output:\n{raw_output}\n\n"
            f"Error: {e}"
        )

    return {
        "plan": validated_plan,
        "current_step_idx": 0
    }
