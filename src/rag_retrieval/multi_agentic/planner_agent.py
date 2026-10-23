"""Planner Agent - multi-agentic-rag's classification step. One
structured-output LLM call, same ChatOpenAI-direct pattern as
single_agentic/orchestration_agent.py. Ported from hrb_chatbot_service's
ai/agents/workflow_agents/planner_agent.py."""

import asyncio
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.search_pre_processing.llm_helpers import AGENT_LLM_TIMEOUT_SECONDS, build_agent_llm
from rag_retrieval.search_pre_processing.prompts import PLANNER_SYSTEM_PROMPT

logger = get_logger("multi_agentic.planner_agent")

AgentName = Literal["vector_kb_agent", "lms_ops_agent", "sql_db_agent", "lms_analytics_agent", "web_search_agent"]


class PlannerTask(BaseModel):
    agent: AgentName
    focus: str
    reasoning: str


class PlannerOutput(BaseModel):
    tasks: list[PlannerTask]


def _build_llm(planner_model: str | None = None) -> ChatOpenAI:
    # Classification is the simplest of this project's agent LLM calls -
    # cheap enough to tier onto its own, overridable model without
    # touching Reviewer/orchestration.
    return build_agent_llm("OPENAI_PLANNER_MODEL", model_override=planner_model)


async def plan(query: str, request_id: str | None = None, planner_model: str | None = None) -> tuple[list[dict], dict]:
    """Classifies one query into one or more {agent, focus} tasks. Never returns an empty list."""
    # include_raw=True exposes the raw AIMessage's usage_metadata -
    # with_structured_output() alone returns only the parsed object.
    llm = _build_llm(planner_model).with_structured_output(PlannerOutput, include_raw=True)
    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=query)]
    config = {"metadata": {"request_id": request_id}} if request_id else None

    started_at = time.perf_counter()
    with log_backend_call(logger, "planner_agent", "plan", request_id=request_id):
        result = await asyncio.wait_for(llm.ainvoke(messages, config=config), timeout=AGENT_LLM_TIMEOUT_SECONDS)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

    output: PlannerOutput | None = result["parsed"]
    tasks = (
        [{"agent": task.agent, "focus": task.focus, "reasoning": task.reasoning} for task in output.tasks]
        if output
        else []
    )
    if not tasks:
        tasks = [
            {
                "agent": "vector_kb_agent",
                "focus": query,
                "reasoning": "Fallback - the Planner's structured output didn't parse, so this "
                "defaulted to the knowledge base rather than a real Planner decision.",
            }
        ]

    logger.info("Planner routed query to %s", [task["agent"] for task in tasks])
    turn = {
        "label": "planner",
        "latency_ms": latency_ms,
        "system_prompt": PLANNER_SYSTEM_PROMPT,
        "human_message": query,
        "chat_history": [],
        "response": str(output) if output else (result["raw"].content or ""),
    }
    return tasks, turn
