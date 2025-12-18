from langgraph.graph import StateGraph, END
from state.research_state import ResearchState

from agents import (
    clarity_scorer,
    clarifier,
    planner,
    supervisor,
    executor,
    report_generator,
)


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("clarity_scorer", clarity_scorer)
    graph.add_node("clarifier", clarifier)
    graph.add_node("planner", planner)
    graph.add_node("supervisor", supervisor)
    graph.add_node("executor", executor)
    graph.add_node("report_generator", report_generator)

    graph.set_entry_point("clarity_scorer")

    graph.add_conditional_edges(
        "clarity_scorer",
        lambda s: "clarifier" if s["clarification_needed"] else "planner",
    )

    graph.add_edge("clarifier", "planner")
    graph.add_edge("planner", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["supervisor_decision"],
        {
            "EXECUTE": "executor",
            "REPLAN": "planner",
            "TERMINATE": "report_generator",
        }
    )

    graph.add_edge("executor", "supervisor")
    graph.add_edge("report_generator", END)

    return graph.compile()
