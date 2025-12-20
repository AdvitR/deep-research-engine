import json
from tests.fakes import FakeLLM
from agents.planner import planner
from state.research_state import ResearchState, PlanStep
from utils.llm import model


def test_planner_initial():
    initial_state: ResearchState = {
        "user_query": "Gather a list of good winter hiking trails in Washington State. Must be within 90 mins from seattle, accessible by car in the winter, between 4-6 miles long, not too dangerous, in the mountains",
        "clarified_query": "Gather a list of good winter hiking trails in Washington State. Must be within 90 mins from seattle, accessible by car in the winter, between 4-6 miles long, not too dangerous, in the mountains",
        "clarity_score": 0.95,
        "clarification_needed": False,
        "research_brief": None,
        "plan": None,
        "current_step_idx": 1,
        "replan_request": None,
        "evidence_store": [],
        "failed_steps": [],
        "supervisor_decision": None,
        "termination_reason": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    updated_fields = planner(initial_state)

    with open("src/data/planner_initial_result.txt", "w", encoding="utf-8") as f:
        print(updated_fields, file=f)

    assert "plan" in updated_fields
    assert isinstance(updated_fields["plan"], list)
    assert len(updated_fields["plan"]) > 0
    for step in updated_fields["plan"]:
        assert isinstance(step, dict)
        assert "id" in step
        assert "goal" in step
        assert "method" in step
        assert "risk" in step
    print("\nTest passed")


def main():
    test_planner_initial()


if __name__ == "__main__":
    main()
