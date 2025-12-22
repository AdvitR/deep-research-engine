from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from utils.llm import get_llm, model
from langchain_core.messages import HumanMessage
from state.research_state import Evidence, ResearchState, PlanStep

# supervisor action space
A_EXECUTE = "EXECUTE"
A_RETRY = "RETRY"
A_SKIP = "SKIP"
A_REPLAN = "REPLAN"
A_TERMINATE = "TERMINATE"

ALLOWED_ACTIONS = {A_EXECUTE, A_RETRY, A_SKIP, A_REPLAN, A_TERMINATE}

# can move to config later
MAX_RETRIES_PER_STEP = 2


# helpers
def _get_query(state: ResearchState) -> str:
    return state.get("clarified_query") or state["user_query"]


def _plan_finished(state: ResearchState) -> bool:
    plan = state.get("plan", [])
    idx = int(state.get("current_step_idx", 0) or 0)
    return idx >= len(plan)


def _get_current_step(state: ResearchState) -> Optional[PlanStep]:
    plan = state.get("plan", [])
    idx = int(state.get("current_step_idx", 0) or 0)
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _failure_counts_by_step(state: ResearchState) -> Counter:
    failed_steps = state.get("failed_steps") or []
    return Counter([f.get("step_id") for f in failed_steps if isinstance(f, dict)])


def _current_step_failures(state: ResearchState) -> List[dict]:
    step = _get_current_step(state)
    if not step:
        return []
    step_id = step["id"]
    return [f for f in (state.get("failed_steps") or []) if f.get("step_id") == step_id]


def _latest_failure_reason_for_step(state: ResearchState, step_id: str) -> str:
    failures = [f for f in state.get("failed_steps", []) if f.get("step_id") == step_id]
    if not failures:
        return "Unknown failure"
    return failures[-1].get("reason", "Unknown failure")


def _retry_budget_exhausted(state, max_retries_per_step: int) -> bool:
    step = _get_current_step(state)
    if not step:
        return True
    counts = _failure_counts_by_step(state)
    return counts.get(step["id"], 0) >= max_retries_per_step


def _replan_budget_exhausted(state: ResearchState) -> bool:
    return int(state.get("replan_count", 0) or 0) >= int(state.get("max_replans", 0) or 0)


def _summarize_failures_for_prompt(state, max_items: int = 3) -> str:
    failures = _current_step_failures(state)
    if not failures:
        return "None"
    lines = []
    for f in failures[-max_items:]:
        reason = (f.get("reason") or "").strip()
        if not reason:
            reason = "Unknown reason"
        lines.append(f"- {reason}")
    return "\n".join(lines)


def _summarize_plan_for_prompt(state, max_steps: int = 6) -> str:
    plan = state.get("plan") or []
    idx = int(state.get("current_step_idx", 0) or 0)

    if not plan:
        return "No plan."

    start = max(0, idx)
    end = min(len(plan), idx + max_steps)

    chunk = plan[start:end]
    out = []
    for i, s in enumerate(chunk, start=start):
        out.append(f"{i}. {s['id']} | {s['method']} | " f"risk={s['risk']} | goal={s['goal']}")
    if end < len(plan):
        out.append(f"... ({len(plan) - end} more steps)")
    return "\n".join(out)


def _normalize_action(raw: str) -> str:
    # accept minor formatting noise
    a = (raw or "").strip().upper()
    # handle cases like `"EXECUTE"` or `EXECUTE\n`
    a = a.strip('"').strip("'").strip()
    # sometimes model might return "Action: EXECUTE"
    if ":" in a:
        a = a.split(":")[-1].strip()
    # sometimes model might return a sentence, keep only first token
    a = a.split()[0] if a else ""
    return a


def _fallback_policy(state, max_retries_per_step: int) -> Tuple[str, str]:
    """
    Returns (action, reason). Used when LLM output is invalid
    or we choose to bypass LLM.
    """
    if _plan_finished(state):
        return (A_TERMINATE, "Plan finished")

    step = _get_current_step(state)
    if step is None:
        return (A_TERMINATE, "No valid current step")

    # if current step has failures, decide between RETRY/REPLAN/SKIP
    failures = _current_step_failures(state)
    if failures:
        if not _retry_budget_exhausted(state, max_retries_per_step):
            return (A_RETRY, "Recent failure; retry budget available")
        if not _replan_budget_exhausted(state):
            return (
                A_REPLAN,
                "Retry budget exhausted; replan budget available",
            )
        # last resort: skip high-risk steps first; otherwise terminate
        if step.get("risk") == "high":
            return (A_SKIP, "Budgets exhausted; high-risk step is skippable")
        return (A_TERMINATE, "Budgets exhausted; cannot progress safely")

    # No failures -> execute
    return (A_EXECUTE, "No failures for current step")


def _llm_decide_action(state: ResearchState, max_retries_per_step: int) -> str:
    """
    Ask the LLM to return ONLY one action token from ALLOWED_ACTIONS.
    """
    llm = get_llm()

    query = _get_query(state)
    step = _get_current_step(state)
    idx = int(state.get("current_step_idx", 0) or 0)

    plan_summary = _summarize_plan_for_prompt(state)
    failure_summary = _summarize_failures_for_prompt(state)

    replan_count = int(state.get("replan_count", 0) or 0)
    max_replans = int(state.get("max_replans", 0) or 0)
    counts = _failure_counts_by_step(state)
    step_id = step["id"] if step else "N/A"
    retries_used = int(counts.get(step_id, 0))

    # Deterministic constraints the LLM must respect
    constraints = []
    if _plan_finished(state):
        constraints.append("Plan is finished => must return TERMINATE.")
    if _replan_budget_exhausted(state):
        constraints.append("Replan budget exhausted => must NOT return REPLAN.")
    if _retry_budget_exhausted(state, max_retries_per_step):
        constraints.append("Retry budget exhausted for current step => must NOT return RETRY.")

    constraints_block = "\n".join(f"- {c}" for c in constraints) if constraints else "- None"

    current_step_line = (
        f"{step['id']} | method={step['method']} | " f"risk={step['risk']} | goal={step['goal']}"
        if step
        else "None"
    )

    prompt = f"""
    You are the SUPERVISOR in a multi-agent research system.

    Your job: decide the SINGLE next action given the current state.

    You must output ONLY ONE of these tokens (no punctuation, no explanation):
    EXECUTE
    RETRY
    SKIP
    REPLAN
    TERMINATE

    Definitions:
    - EXECUTE: run the current plan step as-is.
    - RETRY: retry the current plan step with a modified search query or angle.
    - SKIP: skip the current plan step and move on.
    - REPLAN: ask the planner to revise remaining plan given failures so far.
    - TERMINATE: stop & proceed to report generation with best-effort evidence.

    Research query:
    {query}

    Current step index: {idx}
    Current step:
    {current_step_line}

    Upcoming plan (from current index):
    {plan_summary}

    Failures for current step (most recent last):
    {failure_summary}

    Budgets:
    - retries used for current step: {retries_used} (max{max_retries_per_step})
    - replan_count: {replan_count} (max {max_replans})

    Hard constraints you must respect:
    {constraints_block}

    Decision rules of thumb:
    - If you can likely fix failure by changing search phrasing/scope => RETRY.
    - If the plan is structurally wrong or missing needed steps => REPLAN.
    - If the step is high-risk and blocking progress, and evidence is still
    sufficient overall => SKIP.
    - If steps remain and no blockers => EXECUTE.
    - If plan is done or further progress is impossible => TERMINATE.

    Return ONLY the action token.
    """.strip()

    raw = llm.invoke(prompt).content
    return _normalize_action(raw)


def expand_goal_with_entities(
    goal: str,
    required_entities: List[str],
    entities: Dict[str, List[str]],
) -> str:
    """
    Expand a plan step goal using accumulated entities.
    """

    if not required_entities:
        return goal

    missing = [et for et in required_entities if et not in entities or not entities[et]]
    if missing:
        raise ValueError(f"Cannot expand goal; missing required entities: {missing}")

    context_blocks = []

    for entity_type in required_entities:
        items = entities[entity_type]

        block = f"Context for entity type {entity_type}:\n"
        for i, item in enumerate(items, start=1):
            block += f"{i}. {item}\n"

        context_blocks.append(block)

    expanded_goal = (
        f"{goal.strip()}\n\n"
        f"Use the following context when executing this step:\n\n" + "\n\n".join(context_blocks)
    )

    return expanded_goal


def supervisor(state: ResearchState) -> dict:
    print("=== Supervisor Agent ===")
    """
    Returns a dict update containing at minimum:
      - supervisor_decision: one of ALLOWED_ACTIONS

    Also updates relevant control state:
      - replan_count (increment on REPLAN)
      - current_step_idx (increment on SKIP)
      - termination_reason (set on TERMINATE)
    """
    max_retries_per_step = int(
        state.get(
            "max_retries_per_step",
            MAX_RETRIES_PER_STEP,
        )
        or MAX_RETRIES_PER_STEP
    )

    # deterministic guards
    plan = state.get("plan") or []
    if not isinstance(plan, list) or len(plan) == 0:
        return {
            "supervisor_decision": A_TERMINATE,
            "termination_reason": "No plan available to execute",
        }

    if _plan_finished(state):
        # if plan is finished, terminate gracefully.
        return {"supervisor_decision": A_TERMINATE, "termination_reason": "Plan completed"}

    # LLM-based decision (with validation + fallback)
    try:
        action = _llm_decide_action(state, max_retries_per_step)
    except Exception:
        # if the LLM call fails for any reason, fall back deterministically
        action, _ = _fallback_policy(state, max_retries_per_step)

    if action not in ALLOWED_ACTIONS:
        # invalid LLM output => fallback
        action, _ = _fallback_policy(state, max_retries_per_step)

    # enforce hard constraints even if LLM ignores them
    if action == A_REPLAN and _replan_budget_exhausted(state):
        # choose best alternative
        action, _ = _fallback_policy(state, max_retries_per_step)

    if action == A_RETRY and _retry_budget_exhausted(state, max_retries_per_step):
        action, _ = _fallback_policy(state, max_retries_per_step)

    # apply state updates based on the chosen action
    updates: Dict[str, object] = {"supervisor_decision": action}

    if action == A_REPLAN:
        step = _get_current_step(state)
        failure_reason = (
            _latest_failure_reason_for_step(state, step["id"]) if step else "Unknown failure"
        )

        updates["replan_count"] = int(state.get("replan_count", 0)) + 1
        updates["replan_request"] = {
            "failed_step_id": step["id"] if step else None,
            "failure_reason": failure_reason,
            "current_step_idx": int(state.get("current_step_idx", 0) or 0),
        }

    elif action == A_SKIP:
        updates["current_step_idx"] = int(state.get("current_step_idx", 0)) + 1

    elif action == A_TERMINATE:
        # if we reached here, termination is due to policy choice / fallback.
        updates["termination_reason"] = (
            state.get("termination_reason") or "Supervisor terminated execution"
        )
    # action is EXECUTE

    expanded_goal = expand_goal_with_entities(
        goal=_get_current_step(state)["goal"],
        required_entities=_get_current_step(state).get("requires_entities") or [],
        entities=state.get("entities") or {},
    )
    updates["plan"] = list(state["plan"])  # shallow copy
    updates["plan"][state["current_step_idx"]]["expanded_goal"] = expanded_goal

    print("=== Supervisor Result ===")
    print(updates)
    return updates
