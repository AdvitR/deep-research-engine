import asyncio
from pprint import pprint
from src.graph.main_graph import build_graph

graph = build_graph()

initial_state = {
    "user_query": "Find a major claim from 2020–2022 that is still repeated today but has been partially weakened or reframed by later evidence. Show how the language changed over time and why the earlier framing persists.",
    "clarified_query": None,
    "clarity_score": 0.0,
    "clarification_needed": True,  # or False to skip clarifier
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

final_state = asyncio.run(graph.ainvoke(initial_state))
pprint("Final State:")
pprint(final_state)
