"""genai-rag's own decomposition-merge step, plus generate_follow_ups() -
ported from hrb_chatbot_service's ai/agents/workflow_agents/reviewer_agent.py.
review() is genai-rag's own dependency (Phase 1); generate_follow_ups() is
multi-agentic-rag's (Phase 3) - both live in the one original file, kept
together here too rather than split across two modules."""

import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.core.response_format import TABULAR_FORMAT_INSTRUCTION, should_use_tabular_format
from rag_retrieval.search_pre_processing.context_builder import build_context_from_agent_results
from rag_retrieval.search_pre_processing.llm_helpers import AGENT_LLM_TIMEOUT_SECONDS, build_agent_llm
from rag_retrieval.search_pre_processing.prompts import FOLLOW_UP_QUESTIONS_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT

logger = get_logger("genai.reviewer")


async def review(query: str, agent_results: list[dict], request_id: str | None = None) -> tuple[str, dict | None]:
    """Merges every sub-question's own answer into one final answer."""
    if not agent_results:
        return "I wasn't able to find an answer to that question.", None

    combined = build_context_from_agent_results(agent_results)
    llm = build_agent_llm()
    system_prompt = REVIEWER_SYSTEM_PROMPT
    if should_use_tabular_format(query):
        system_prompt = f"{REVIEWER_SYSTEM_PROMPT}\n\n{TABULAR_FORMAT_INSTRUCTION}"
    human_message = f"Original question: {query}\n\nDomain agent results:\n{combined}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]

    config = {"metadata": {"request_id": request_id}} if request_id else None
    started_at = time.perf_counter()
    with log_backend_call(logger, "reviewer", "review", request_id=request_id):
        response = await asyncio.wait_for(llm.ainvoke(messages, config=config), timeout=AGENT_LLM_TIMEOUT_SECONDS)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

    turn = {
        "label": "reviewer",
        "latency_ms": latency_ms,
        "system_prompt": system_prompt,
        "human_message": human_message,
        "chat_history": [],
        "response": response.content,
    }
    return response.content, turn


async def generate_follow_ups(query: str, answer: str) -> list[str]:
    """2-3 suggested follow-up questions - a nice-to-have, so any failure
    here returns [] rather than failing the whole request."""
    llm = build_agent_llm()
    messages = [
        SystemMessage(content=FOLLOW_UP_QUESTIONS_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nAnswer: {answer}"),
    ]

    try:
        with log_backend_call(logger, "reviewer", "generate_follow_ups"):
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=AGENT_LLM_TIMEOUT_SECONDS)
    except Exception as error:
        logger.warning("Follow-up question generation failed: %s: %s", type(error).__name__, error)
        return []

    return [line.strip() for line in response.content.splitlines() if line.strip()]
