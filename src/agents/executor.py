import json
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from dotenv import load_dotenv
from tavily import TavilyClient
from utils.llm import model
from state.research_state import Evidence, ResearchState, PlanStep
from langchain_core.messages import HumanMessage
from utils.tavily_wrapper import tavily_search, tavily_extract


def decompose_plan_step(step: str, entity_context: dict, prev_err: str | None) -> List[str]:
    """Decomposes a plan step into a list of subtasks."""
    prompt = f"""
You are a domain-aware research assistant. Your task is to decompose the following high-level research step into a minimal set of **atomic**, **web-searchable** subtasks.

### Guidelines:
- Each subtask should be **a concise query** that could be entered into a search engine to gather factual, relevant information.
- Do **not** reference specific documents, websites, or named authors unless they are widely known entities (e.g., Wikipedia, WHO, NASA).
- Avoid subtasks that are too vague (e.g., "learn about X") or too narrow (e.g., "read section 4.1 of the 2017 IMF report").
- Do **not** generate subtasks that involve clarification, introspection, or LLM-only reasoning (e.g., “determine if X is unclear” or “summarize findings”).
- Use neutral phrasing, and focus on **fact-finding**, **comparisons**, **definitions**, **statistics**, or **causal relationships**.
- Include only as many subtasks as are **necessary** to cover the plan step comprehensively.
- There may be a list of entities extracted from prior steps that can help provide context for this step. Use them if relevant. You can make one subtask for each entity if required.

### Plan Step:
"{step}"

### Previous Errors:
{prev_err if prev_err else "None"}

### Output Format:
Return only the subtasks as a numbered list.
Each item should be a single-line query.
"""
    print("DECOMPOSE PROMPT\n", prompt)
    response = model.invoke([HumanMessage(content=prompt)]).content
    return [line.strip().split(". ", 1)[-1] for line in response.splitlines() if line.strip()]


def shorten_plan_subtask(subtask: str, limit: int) -> str:
    """Shortens a plan subtask to a specified limit."""
    if len(subtask) <= limit:
        return subtask

    prompt = f"""
Shorten the following sentence to under {limit} characters while preserving its meaning and specificity:

"{subtask}"
"""
    response = model.invoke([HumanMessage(content=prompt)]).content
    return response


def choose_best_n_urls(subtask: str, urls: List[str], n: int) -> List[int]:
    """Chooses the best N URLs from a list of URLs based on a given subtask."""
    prompt = f"""
You're evaluating URLs for relevance to the following research subtask:
"{subtask}"

Given these URLs:
{chr(10).join([f"{i+1}. {url}" for i, url in enumerate(urls)])}

Return the numbers of the {n} most relevant URLs in order of usefulness. The URLS should be crawlable, so exclude sites like Reddit, or PDFs, or other non-crawlable content.
Just return a comma-separated list of numbers (e.g., 2,1,5).
"""
    response = model.invoke([HumanMessage(content=prompt)]).content
    try:
        indexes = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
        return indexes[:n]
    except Exception:
        return [i for i in range(n)]


def extract_info_from_page(subtask: str, page_content: str) -> str:
    """Extracts information from a page's content that is relevant to the subtask. Gets rid of unnecessary things"""
    prompt = f"""
You are an information extraction agent. Your task is to extract only factual, relevant content from the following web page, based on this research subtask:

Subtask:
"{subtask}"

### Instructions:
- Keep only the sections that are directly relevant to the subtask.
- Make sure to extract numeric data if it appears.
- Exclude boilerplate elements like navigation menus, ads, author bios, prompts to subscribe, cookie notices, or unrelated sections.
- Ignore links, images, citations, and formatting — focus on the core informative content.
- The output should be a clean, readable summary of the key factual information related to the subtask.
- Do not hallucinate or add any extra context or information not found in the content.

### Page Content:
\"\"\"
{page_content}
\"\"\"

### Cleaned Output:
"""
    return model.invoke([HumanMessage(content=prompt)]).content.strip()


def evaluate_subtask_result(subtask: str, result: str) -> int:
    """Evaluates the result of a subtask. Returns a score between 0 and 10"""
    prompt = f"""
Evaluate the relevance and quality of the following result for this research subtask:

Subtask:
"{subtask}"

Result:
\"\"\"
{result}
\"\"\"

Score the result on a scale from **0 to 10** based on the following dimensions:

- **Relevance**: Does the result directly address the subtask?
- **Completeness**: Does it substantially cover the information needed to answer the subtask?
- **Factual Quality**: Is the information specific, concrete, and plausibly reliable (not vague or speculative)?

### Scoring Guidance:
- **9-10**: Directly answers the subtask with clear, detailed, and relevant factual information.
- **6-8**: Mostly relevant and useful, but missing some important details or depth.
- **3-5**: Partially related, superficial, or only indirectly useful.
- **1-2**: Barely related or mostly noise.
- **0**: Irrelevant, incorrect, or empty.

### Output Requirements:
- Return **only a single integer** between 0 and 10.
- Do not include explanations, text, or formatting.
"""
    response = model.invoke([HumanMessage(content=prompt)]).content
    try:
        return min(max(int(response.strip()), 0), 10)
    except Exception:
        return 5  # fallback neutral


def _extract_entities_from_text(
    expected_entities: List[str], evidence_text: str
) -> Dict[str, List[str]]:
    """
    Extract entities of specified types from raw evidence text.

    Returns a dict keyed by entity type, each mapping to a list of strings.
    """

    if not expected_entities or not evidence_text.strip():
        return {entity: [] for entity in expected_entities}

    prompt = f"""
You are an information extraction system.

Your task is to extract structured entities from the text below.

ENTITY TYPES TO EXTRACT:
{expected_entities}

RULES:
- Only extract entities of the specified types.
- If an entity is not present, return an empty list for that type.
- Each entity must be a string.
- Do NOT hallucinate.
- Do NOT include explanations.

OUTPUT FORMAT (STRICT JSON):
{{
  "<entity_type>": [{{ ... }}, {{ ... }}],
  ...
}}

TEXT:
{evidence_text}
"""

    response = model.invoke([HumanMessage(content=prompt)]).content

    try:
        parsed = json.loads(response)
    except Exception:
        parsed = {entity: [] for entity in expected_entities}

    # Normalize output
    result: Dict[str, List[str]] = {}
    for entity in expected_entities:
        values = parsed.get(entity, [])
        if isinstance(values, list):
            result[entity] = [str(v) for v in values]
        else:
            result[entity] = []
    return result


def _extract_entities(step: PlanStep, evidence: List[Evidence]) -> List[str]:
    """
    Extract entities from the plan step goal for better context.
    This is a placeholder function and should be replaced with a proper NER model.
    """
    expected_entities = step.get("produces_entities", [])
    aggregated: Dict[str, List[str]] = defaultdict(list)

    if not expected_entities:
        return {}

    for ev in evidence:
        extracted = _extract_entities_from_text(expected_entities, ev)

        for entity_type, values in extracted.items():
            aggregated[entity_type] += values

    for entity_type, values in aggregated.items():
        seen = set()
        deduped = []
        for v in values:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        aggregated[entity_type] = deduped

    return aggregated


def execute_subtask(subtask: str) -> str:
    shortened_subtask = shorten_plan_subtask(subtask, limit=400)
    print("SHORTENED", shortened_subtask)
    search_response = tavily_search(shortened_subtask)
    urls = [result["url"] for result in search_response["results"]]
    best_url_indexes = choose_best_n_urls(subtask, urls, n=3)
    best_urls = [urls[i] for i in best_url_indexes]
    print("BEST URLs", best_urls)

    page_content = tavily_extract(best_urls)
    best_extracted_info = None
    max_score = -1
    for result in page_content["results"]:
        # Process the extracted content
        content = result["raw_content"]
        url = result["url"]
        print("CONTENT", content[:100])
        extracted_info = extract_info_from_page(subtask, content)
        print("EXTRACTED FROM", url, ":", extracted_info)
        score = evaluate_subtask_result(subtask, extracted_info)
        if score > max_score or best_extracted_info is None:
            max_score = score
            best_extracted_info = extracted_info

    return best_extracted_info


def executor(state: ResearchState) -> dict:
    """
    Returns a list of results for each subtask in a single plan step.

    :param state: Description
    :type state: ResearchState
    :return: Description
    :rtype: dict
    """

    step_idx = state["current_step_idx"]
    step_goal = state["plan"][step_idx]["expanded_goal"]
    prev_err = (
        state["failed_steps"][step_idx]["reason"] if step_idx in state["failed_steps"] else None
    )
    entity_context = state.get("entities", {})
    required_entities = state["plan"][step_idx].get("requires_entities", [])
    entity_context = {k: v for k, v in entity_context.items() if k in required_entities}
    print("ENTITY CONTEXT", entity_context)
    subtask_list = decompose_plan_step(step_goal, entity_context, prev_err)
    print("SUBTASKS", subtask_list)
    subtask_results = []
    for subtask in subtask_list:
        result = execute_subtask(subtask)
        subtask_results.append(result)

    new_entities = _extract_entities(state["plan"][step_idx], subtask_results)

    merged_entities = state.get("entities", {})
    for entity_type, values in new_entities.items():
        merged_entities.setdefault(entity_type, [])
        merged_entities[entity_type].extend(
            v for v in values if v not in merged_entities[entity_type]
        )

    evidence_store = list(state.get("evidence_store", []))
    while len(evidence_store) <= step_idx:
        evidence_store.append([])
    evidence_store[step_idx] = state["evidence_store"][step_idx]

    return {
        "evidence_store": evidence_store,
        "entities": merged_entities,
    }
