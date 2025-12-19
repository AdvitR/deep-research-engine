from typing import List, Dict
from agents.executor import executor
from state.research_state import PlanStep, ResearchState
import os
from dotenv import load_dotenv
from tavily.client import TavilyClient
from utils.tavily_wrapper import tavily_search

response = tavily_search("NHS Outcomes Framework 2023/24 indicators list NHS England")
print(response)
