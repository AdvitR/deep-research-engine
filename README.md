# Deep Research Engine (LangGraph-Based)

A learning-oriented implementation of a Deep Research System built using LangGraph and LLM-based agents. The system is designed to answer complex research queries that require external information retrieval, multi-step reasoning, and robust failure handling, rather than single-shot generation.

A representative example of the type of query this system targets is:

> “Compare the health outcomes and cost-efficiency of the UK’s NHS and Germany’s health insurance system using recent data.”

Such queries cannot be answered reliably without searching external sources, aggregating information across multiple steps, and adapting when some data is missing or infeasible to obtain. This project focuses on building an agent architecture that can do exactly that.

## Architectural Overview

The system is organized around a shared mutable state (`ResearchState`) and a set of agents that read from and write to that state. Control flow between agents is managed by LangGraph, forming a cyclic graph rather than a DAG pipeline.

The system consists of the following agents:

### Clarity scoring and clarification

The system's `clarity_scorer` first evaluates how well-specified the user’s query is by providing a clarity score between `0` and `1`. If the query is deemed insufficiently clear, indicated by a clarity score below a configurable threshold (`0.6` in the current setup), the pipeline triggers the `clarifier` agent. The clarifier asks targeted follow-up questions aimed at resolving ambiguities (e.g., scope, timeframe, geography). If the clarity score exceeds the threshold, the clarification step is skipped entirely and the system proceeds directly to planning.

### Planner

The `planner` agent converts the (possibly clarified) user query into a small, executable research plan. Each plan consists of an ordered list of steps, where each step represents a single, concrete information objective such as retrieving a specific class of metrics or performing a simple aggregation. Steps are annotated with a method (`search` or `analysis`) and a coarse risk level indicating expected data availability.

The planner supports two modes of operation:

*   **Initial planning**: where a complete plan is generated from scratch.

*   **Scoped replanning**: which is triggered when execution fails for a specific step. In this mode, the planner preserves all completed steps and replaces only the remaining portion of the plan, explicitly avoiding repetition of the failed step.

Planner outputs are strictly validated before being accepted into state to ensure downstream safety.

### Supervisor

The supervisor acts as the system’s control policy. At each iteration, it inspects the current shared state and decides the next action to take. The available actions are:

- `EXECUTE`: run the current plan step as-is

- `RETRY`: retry the current step with a modified search strategy

- `SKIP`: skip the current step and advance the plan

- `REPLAN`: request the planner to revise the remaining steps

- `TERMINATE`: stop execution and proceed to report generation

The supervisor combines deterministic guards (retry budgets, replan budgets, plan completion checks) with an optional LLM-based decision prompt. Deterministic fallbacks are enforced whenever the LLM output is invalid or violates hard constraints. This design ensures that control flow remains safe and predictable even when model outputs are unreliable.

### Executor 

The `executor` is responsible for carrying out individual plan steps. Each plan step is treated as a high-level information objective, not a single atomic action. Hence, executing a step usually involves multiple subtasks including generating search queries, performing multiple web searches, filtering sources, and aggregating evidence.

The executor evaluates whether a step succeeded based on configurable criteria (e.g., minimum amount and confidence of evidence) and records failures without making any control-flow decisions itself.

### Replanning and failure handling

Failures during execution are treated as first-class data rather than terminal errors. When a step fails repeatedly or is deemed structurally infeasible, the supervisor can request a scoped replan. During replanning, failures associated with replaced steps are explicitly cleared from state to prevent stale error propagation.

This approach allows the system to adapt to missing or unavailable data while preserving progress made on earlier steps, rather than restarting the entire pipeline.

### Report Generator 

The `report_generator` is the final stage of the pipeline. It synthesizes the collected evidence, the original research query, and any unknown gaps into a single final report. The report generator uses only the evidence stored in state and does not perform additional retrieval or reasoning steps.

If the system terminates early due to exhausted budgets or infeasible steps, the report generator produces a best-effort partial answer and explicitly documents limitations and missing information. The final output is stored in the shared state under the `final_report` field. 

### Shared state and execution model

All agents operate over a shared `ResearchState` object, which is incrementally updated as the system progresses. This state includes the research query, plan, execution pointer, evidence store, failure records, replanning metadata, and final report. LangGraph is used to organize control flow between agents based on supervisor decisions.

