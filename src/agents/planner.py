import json
from typing import List
from state.research_state import ResearchState, PlanStep
from langchain_core.messages import HumanMessage
from utils.llm import model

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
        required_keys = {"id", "goal", "method", "risk", "produces_entities", "requires_entities"}
        if set(step.keys()) != required_keys:
            raise ValueError(f"Step {i} keys must be {required_keys}, got {step.keys()}")

        # asserting types and values for all required keys
        if not isinstance(step["id"], str):
            raise ValueError(f"Step {i} id must be a string")

        if not isinstance(step["goal"], str) or not step["goal"].strip():
            raise ValueError(f"Step {i} goal must be a non-empty string")

        if step["method"] not in ALLOWED_METHODS:
            raise ValueError(f"Invalid method in step {i}")

        if step["risk"] not in ALLOWED_RISKS:
            raise ValueError(f"Invalid risk in step {i}")

        if not isinstance(step["produces_entities"], list) or not all(
            isinstance(e, str) for e in step["produces_entities"]
        ):
            raise ValueError(f"Step {i} produces_entities must be a list of strings")

        validated.append(step)

    if len(validated) == 0:
        raise ValueError("Planner returned an empty plan")

    return validated


def planner(state: ResearchState) -> dict:
    print("=== Planner Agent ===")
    replan_request = state.get("replan_request")

    query = state.get("clarified_query") or state["user_query"]

    # INITIAL PLANNING
    if replan_request is None:
        prompt = f"""
        You are a research planning agent in a multi-agent system.

        Your job is to convert a high-level research query into a SMALL,
        EXECUTABLE sequence of research steps that can be carried out using
        external information sources (e.g., web search, reports, datasets).

        Research Query:
        {query}

        Planning Objectives:
        - Decompose the query into concrete, factual sub-questions.
        - Each step should aim to retrieve or analyze real-world information.
        - Prefer steps that can be answered using public, authoritative sources
        (e.g, government reports, OECD, World Bank, peer-reviewed surveys).
        - Avoid steps that rely on speculation, private data, or judgment.

        Step Design Rules:
        For EACH step:
        1. The step must be independently executable.
        2. The step must correspond to ONE clear information need.
        3. The step must be feasible via search or simple analysis.
        4. If data availability is uncertain, mark the step as higher risk.
        5. Each step should just be a search for information. Don't include any analysis steps in the initial plan.

        Risk Levels:
        - "low": Data is very likely to exist in public sources.
        - "medium": Data likely exists but may require synthesis or proxies.
        - "high": Data may be incomplete, outdated, or unavailable.

        OUTPUT FORMAT (STRICT)
        Return ONLY a JSON list (no markdown, no explanation).

        Each list element must have EXACTLY these fields:
        - "id": short string identifier (e.g., "s1", "s2")
        - "goal": precise description of what the step aims to find or compute
        - "method": one of ["search", "analysis"]
        - "risk": one of ["low", "medium", "high"]
        - "produces_entities": list of strings representing entities produced by this step. Each entity name should fully describe the data produced (e.g., "average_annual_rainfall_by_country"). This list can be empty.
        - "requires_entities": list of strings representing entities required by this step. The name should match those produced by prior steps. This list can be empty.

        IMPORTANT CONSTRAINTS:
        - Do NOT answer the research question.
        - Do NOT include conclusions.
        - Do NOT include explanatory text.
        - Focus ONLY on planning executable steps.
        - Use the minimum number of steps required to answer the query well.

        Begin.
        """.strip()

        raw_output = model.invoke([HumanMessage(content=prompt)]).content

        try:
            parsed = json.loads(raw_output)
            validated_plan = _validate_plan(parsed)
        except Exception as e:
            raise RuntimeError(
                f"Planner failed to produce valid output.\n"
                f"Raw output:\n{raw_output}\n\n"
                f"Error: {e}"
            )

        print("=== Planner Result ===")
        print({"plan": validated_plan, "current_step_idx": 0})
        return {"plan": validated_plan, "current_step_idx": 0}

    # SCOPED REPLANNING
    else:
        failed_step_id = replan_request["failed_step_id"]
        failure_reason = replan_request["failure_reason"]
        k = int(replan_request["current_step_idx"])
        old_plan = state["plan"]
        completed_steps = old_plan[:k]

        prompt = f"""
        You are a research planning agent in a multi-agent system.

        Your job is to REVISE an existing research plan after a specific
        failure, by proposing a NEW, EXECUTABLE sequence of steps that replaces
        ONLY the failed portion of the plan.

        Research Query:
        {query}

        Context:
        - The plan was partially executed successfully.
        - A specific step FAILED and must NOT be repeated.
        - Steps completed before failure are correct and MUST remain unchanged.

        Failure Details:
        - Failed Step ID: {failed_step_id}
        - Failure Reason: {failure_reason}
        - Failure occurred at plan index: {k}

        Completed Steps (DO NOT MODIFY):
        {completed_steps}

        Replanning Objectives:
        - Replace ONLY the remaining steps starting at index {k}.
        - Avoid repeating the failed step or closely equivalent steps.
        - Adapt the plan to bypass missing, unavailable, or infeasible data.
        - Preserve the original intent of answering the research query.
        - Prefer alternative metrics, proxies, or authoritative sources.

        Step Design Rules:
        For EACH new step:
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
        - "goal": precise description of what the step aims to find or compute
        - "method": one of ["search", "analysis"]
        - "risk": one of ["low", "medium", "high"]
        - "produces_entities": list of strings representing entities produced by this step. This list can be empty.
        - "requires_entities": list of strings representing entities required by this step. The name should match those produced by prior steps. This list can be empty.

        IMPORTANT CONSTRAINTS:
        - Do NOT include the completed steps above.
        - Do NOT answer the research query.
        - Do NOT include conclusions or explanations.
        - Focus ONLY on proposing replacement executable steps.
        - Use the minimum number of steps required to complete the plan.

        Begin.
        """.strip()

        # TODO: Fix LLM invocation
        raw_output = model.invoke([HumanMessage(content=prompt)]).content

        try:
            parsed = json.loads(raw_output)
            new_steps = _validate_plan(parsed)
        except Exception as e:
            raise RuntimeError(
                f"Planner failed to produce valid output.\n"
                f"Raw output:\n{raw_output}\n\n"
                f"Error: {e}"
            )

        new_plan = completed_steps + new_steps

        # clear failures for replaced steps
        preserved_step_ids = {step["id"] for step in completed_steps}

        filtered_failure_steps = [
            f for f in state.get("failed_steps", []) if f.get("step_id") in preserved_step_ids
        ]

        print("=== Planner Result ===")
        print({"plan": new_plan, "current_step_idx": k})
        return {
            "plan": new_plan,
            "current_step_idx": k,
            "replan_request": None,  # IMPORTANT: clears it
            "failed_steps": filtered_failure_steps,
        }
