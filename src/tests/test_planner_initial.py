import json
from tests.fakes import FakeLLM
import agents.planner as planner_mod


def test_planner_initial(monkeypatch):
    fake_plan = [
        {"id": "s1", "goal": "Find A", "method": "search", "risk": "low"},
        {"id": "s2", "goal": "Find B", "method": "search", "risk": "medium"},
    ]
    monkeypatch.setattr(planner_mod, "get_llm", lambda:
                        FakeLLM(json.dumps(fake_plan)))

    state = {
        "user_query": "Compare X vs Y",
        "clarified_query": None,
        "replan_request": None,
    }

    upd = planner_mod.planner(state)
    assert upd["current_step_idx"] == 0
    assert upd["plan"][0]["id"] == "s1"
