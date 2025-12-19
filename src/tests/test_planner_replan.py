import json
from tests.fakes import FakeLLM
import agents.planner as planner_mod


def test_planner_replan_preserves_prefix_and_clears_failures(monkeypatch):
    # Old plan
    old_plan = [
        {"id": "s1", "goal": "Find A", "method": "search", "risk": "low"},
        {"id": "s2", "goal": "Find B", "method": "search", "risk": "medium"},
        {"id": "s3", "goal": "Find C", "method": "search", "risk": "high"},
    ]

    # New tail returned by LLM (must NOT include s1)
    new_tail = [
        {"id": "s4", "goal": "Alternative for B",
            "method": "search", "risk": "medium"},
        {"id": "s5", "goal": "Alternative for C",
            "method": "search", "risk": "high"},
    ]

    monkeypatch.setattr(planner_mod, "get_llm",
                        lambda: FakeLLM(json.dumps(new_tail)))

    state = {
        "user_query": "Compare X vs Y",
        "clarified_query": None,
        "plan": old_plan,
        "failed_steps": [
            {"step_id": "s2", "reason": "No data"},
            {"step_id": "s3", "reason": "Paywalled"},
            {"step_id": "s1", "reason": "(should be preserved if exists)"},
        ],
        "replan_request": {
            "failed_step_id": "s2",
            "failure_reason": "No data",
            "current_step_idx": 1,  # preserve s1
        }
    }

    upd = planner_mod.planner(state)

    # plan prefix preserved
    assert upd["plan"][0]["id"] == "s1"
    # tail replaced
    assert [s["id"] for s in upd["plan"][1:]] == ["s4", "s5"]
    # index unchanged (k)
    assert upd["current_step_idx"] == 1
    # replan_request cleared
    assert upd["replan_request"] is None
    # failures only for preserved prefix steps
    assert all(f["step_id"] == "s1" for f in upd["failed_steps"])
