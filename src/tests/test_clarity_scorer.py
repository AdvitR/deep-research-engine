from typing import List, Dict
from agents.clarity_scorer import clarity_scorer
from state.research_state import PlanStep, ResearchState


def test_clarity_scorer():
    state: ResearchState = {
        "user_query": "Gather a list of good winter hiking trails in Washington State. Must be within 90 mins from Seattle, accessible by car in the winter, between 4-6 miles long, not too dangerous, in the mountains",
        "clarified_query": None,
        "clarity_score": 0.0,
        "clarification_needed": False,
        "research_brief": None,
        "plan": [],
        "current_step_idx": 0,
        "replan_request": None,
        "evidence_store": [],
        "failed_steps": [],
        "supervisor_decision": None,
        "termination_reason": None,
        "replan_count": 0,
        "max_replans": 3,
    }

    updated_fields = clarity_scorer(state)
    with open("src/data/clarity_scorer_result.txt", "w", encoding="utf-8") as f:
        print(updated_fields, file=f)

    assert "clarity_score" in updated_fields
    assert "clarification_needed" in updated_fields
    assert 0.0 <= updated_fields["clarity_score"] <= 1.0
    print("\nTest passed")


def main():
    test_clarity_scorer()


if __name__ == "__main__":
    main()
