from typing import Dict, List
from utils.llm import get_llm
from state.research_state import ResearchState, Evidence, PlanStep

def _format_evidence_summary(
        plan: List[PlanStep],
        evidence_store: List[Evidence],
        failed_steps: List[Dict],
) -> str:
    """
    Produces a structured summary of evidence grouped by plan step. 
    Assumes all evidence items are tagged with step_id.
    """
    lines = []
    lines.append("EVIDENCE BY PLAN STEP:\n")

    evidence_by_step = {step["id"]: [] for step in plan}
    for ev in evidence_store:
        evidence_by_step.setdefault(ev["step_id"], []).append(ev)

    for step in plan:
        step_id = step["id"]
        lines.append(f"- Step {step_id}: {step['goal']}")

        step_evidence = evidence_by_step.get(step_id, [])
        if step_evidence:
            for ev in step_evidence:
                src = ev.get("source", "unknown source")
                conf = ev.get("confidence", "NA")
                content = ev.get("content", "").strip()
                lines.append(f"    * ({conf}) {content} [{src}]")
        else:
            lines.append("  * No evidence collected for this step.")

    if failed_steps:
        lines.append("\nFAILED / INCOMPLETE STEPS")
        for f in failed_steps:
            step_id = f.get("step_id", "unknown")
            reason = f.get("reason", "no reason provided")
            lines.append(f"- Step {step_id}: {reason}")

    return "\n".join(lines)

def report_generator(state: ResearchState) -> dict:
    llm = get_llm()

    query = state.get("clarified_query") or state["user_query"]
    plan = state["plan"]
    evidence_store = state["evidence_store"]
    failed_steps = state["failed_steps"]
    termination_reason = state.get("termination_reason")

    evidence_summary = _format_evidence_summary(
        plan, evidence_store, failed_steps
    )

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

    If evidence is insufficient to fully answer the question, provide
    a best-effort partial answer and explicitly explain what is missing.

    Collected Evidence:
    {evidence_summary}

    Termination Context:
    {termination_reason or "Normal completion"}

    Write a clear, well-structured report.
    Do NOT mention internal agents, steps, or system details.
    """.strip()

    final_report = llm.involke(prompt).content.strip()

    return {
        "final_report": final_report
    }
