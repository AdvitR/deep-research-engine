from typing import List, Dict
from agents.executor import executor
from state.research_state import PlanStep, ResearchState


def test_executor_agent():
    """Tests the executor agent end-to-end on a simple example plan step."""

    # Define a basic plan with one research goal
    plan: List[PlanStep] = [
        {
            "id": "s1",
            "goal": "Find recent NHS health outcomes",
            "method": "web_search",
            "expected_sources": ["OECD", "UK Gov"],
            "risk": "medium",
        },
        {
            "id": "s2",
            "goal": "Find German system cost-efficiency metrics",
            "method": "web_search",
            "expected_sources": ["OECD", "WHO"],
            "risk": "high",
        },
    ]

    # Construct a minimal initial state
    initial_state: ResearchState = {
        "user_query": "Compare the health outcomes and cost-efficiency of the UK's NHS and Germany's health insurance system using recent data.",
        "clarified_query": "Compare the health outcomes and cost-efficiency of the UK's NHS and Germany's health insurance system using recent data.",
        "clarity_score": 0.95,
        "clarification_needed": False,
        "research_brief": None,
        "plan": plan,
        "current_step_idx": 1,
        "replan_request": None,
        "evidence_store": [],
        "failed_steps": [],
        "supervisor_decision": None,
        "termination_reason": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    # Run the executor
    print("Running executor...")
    updated_fields = executor(initial_state)

    # Print results
    print("=== Executor Test Output ===")
    evidence_list = updated_fields.get("evidence_store", [])
    with open("executor_result.txt", "w", encoding="utf-8") as f:
        print(updated_fields, file=f)
    for i, evidence_group in enumerate(evidence_list):
        print(f"\nStep {i} Evidence:")
        for j, evidence in enumerate(evidence_group):
            print(f"  [{j}]")
            if isinstance(evidence, dict):
                for k, v in evidence.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {evidence}")

    # Optional: basic assertions
    assert len(evidence_list) > 0, "No evidence returned"
    assert isinstance(evidence_list[0], list), "Evidence not grouped by step"
    print("\nTest passed")


def main():
    test_executor_agent()


if __name__ == "__main__":
    main()
