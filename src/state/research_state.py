from typing import TypedDict, List, Optional, Literal


class PlanStep(TypedDict):
    id: str
    goal: str
    method: Literal["search", "analysis"]
    risk: Literal["low", "medium", "high"]


class Evidence(TypedDict):
    source: str
    content: str
    confidence: float


class FailureRecord(TypedDict):
    step_id: str
    reason: str


class ResearchState(TypedDict):
    # user input
    user_query: str
    clarified_query: Optional[str]

    # clarification
    clarity_score: float
    clarification_needed: bool

    # planning
    research_brief: Optional[str]
    plan: List[PlanStep]
    current_step_idx: int
    replan_request: Optional[dict]

    # execution memory
    evidence_store = List[Evidence]
    failed_steps = List[FailureRecord]

    # control
    supervisor_decision: Optional[str]
    termination_reason: Optional[str]

    # loop control
    replan_count: int
    max_replans: int
