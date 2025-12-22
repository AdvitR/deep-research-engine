from typing import Any, Set, TypedDict, List, Optional, Literal


class PlanStep(TypedDict):
    id: str
    goal: str
    expanded_goal: Optional[str]
    method: Literal["search", "analysis"]
    risk: Literal["low", "medium", "high"]
    produces_entities: List[str]
    requires_entities: List[str]


class Evidence(TypedDict):
    step_id: setattr
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
    entities: dict[str, List[str]]
    evidence_store: List[List[str]]
    failed_steps: List[FailureRecord]
    estimate: bool  # Whether to give an estimate of evidence in case it's not findable, just for testing purposes

    # control
    supervisor_decision: Optional[str]
    termination_reason: Optional[str]

    # loop control
    replan_count: int
    max_replans: int

    final_report: Optional[str]
