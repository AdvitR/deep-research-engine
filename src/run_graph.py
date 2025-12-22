import asyncio
from pprint import pprint
from src.graph.main_graph import build_graph

graph = build_graph()

initial_state = {
    "user_query": "Investigate the 2023–2024 U.S. Department of Justice antitrust actions against major technology companies. Identify one specific enforcement action where at least three reputable outlets disagree on the primary motivation or legal theory. Cite the exact statutory language used by DOJ, contrast it with each outlet’s framing, and explain which interpretation is best supported by the complaint text.",
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
