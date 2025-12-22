from typing import Dict, List
from utils.llm import get_llm, model
from langchain_core.messages import HumanMessage
from state.research_state import ResearchState, Evidence, PlanStep


def _format_evidence_summary(
    plan: List[PlanStep],
    evidence_store: List[List[str]],
    failed_steps: List[Dict],
) -> str:
    """
    Produces a structured summary of evidence grouped by plan step.
    Each entry in evidence_store is a list of strings (evidence per step).
    """
    lines = []
    lines.append("EVIDENCE BY PLAN STEP:\n")

    for idx, step in enumerate(plan):
        step_id = step.get("id", f"step-{idx}")
        step_goal = step.get("goal", "[No goal defined]")
        lines.append(f"- Step {step_id}: {step_goal}")

        if idx < len(evidence_store) and evidence_store[idx]:
            for i, ev in enumerate(evidence_store[idx]):
                evidence_snippet = ev.strip().replace("\n", " ")
                lines.append(f"    * {evidence_snippet}")
        else:
            lines.append("    * No evidence collected for this step.")

    if failed_steps:
        lines.append("\nFAILED / INCOMPLETE STEPS")
        for f in failed_steps:
            step_id = f.get("step_id", "unknown")
            reason = f.get("reason", "no reason provided")
            lines.append(f"- Step {step_id}: {reason}")

    return "\n".join(lines)


def report_generator(state: ResearchState) -> dict:
    print("=== Report Generator Agent ===")
    query = state.get("clarified_query") or state["user_query"]
    plan = state["plan"]
    evidence_store = state["evidence_store"]
    failed_steps = state["failed_steps"]
    termination_reason = state.get("termination_reason")

    evidence_summary = _format_evidence_summary(plan, evidence_store, failed_steps)
    # print("=== Evidence Summary ===")
    # print(evidence_summary)

    prompt = f"""
    You are a research assistant writing a final report.

    Your task is to answer the following research question using ONLY
    the evidence provided below.

    Research Question:
    {query}

    You MUST:
    - Directly answer the research question.
    - Base your answer strictly on the provided evidence.
    - Clearly state any uncertainties, missing data, or limitations.
    - Avoid speculation or unsupported claims.
    - Prefer cautious, qualified language where evidence is incomplete.
    - Treat estimated evidence as if it were real.

    If evidence is insufficient to fully answer the question, provide
    a best-effort partial answer and explicitly explain what is missing.

    Collected Evidence:
    {evidence_summary}

    Termination Context:
    {termination_reason or "Normal completion"}

    Write a clear, well-structured report.
    Do NOT mention internal agents, steps, or system details.
    """.strip()

    final_report = model.invoke([HumanMessage(content=prompt)]).content.strip()

    print("=== Report Generator Result ===")
    print({"final_report": final_report})
    return {"final_report": final_report}
