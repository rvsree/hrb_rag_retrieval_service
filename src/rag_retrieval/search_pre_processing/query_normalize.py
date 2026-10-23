"""Pre-retrieval sanity pass - ported from hrb_chatbot_service's
ai/search_pre_processing/query_normalize.py, same structured-output pattern."""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.search_pre_processing.llm_helpers import AGENT_LLM_TIMEOUT_SECONDS, build_agent_llm
from rag_retrieval.search_pre_processing.prompts import NORMALIZE_SYSTEM_PROMPT

logger = get_logger("pre_processing.query_normalize")


class NormalizedQuery(BaseModel):
    normalized_query: str


def _build_llm() -> ChatOpenAI:
    return build_agent_llm("OPENAI_NORMALIZE_MODEL")


async def normalize_query(query: str) -> str:
    """Fixes spelling/grammar only - never changes meaning. Falls back to
    the original query unchanged on any failure."""
    if not query.strip():
        return query

    llm = _build_llm().with_structured_output(NormalizedQuery, include_raw=True)
    messages = [SystemMessage(content=NORMALIZE_SYSTEM_PROMPT), HumanMessage(content=query)]

    try:
        with log_backend_call(logger, "query_normalize", "normalize"):
            result = await asyncio.wait_for(llm.ainvoke(messages), timeout=AGENT_LLM_TIMEOUT_SECONDS)
    except Exception as error:
        logger.warning("Normalization failed for %r, using the original query: %s: %s", query, type(error).__name__, error)
        return query

    output: NormalizedQuery | None = result["parsed"]
    normalized = output.normalized_query.strip() if output else ""
    if not normalized:
        return query

    if normalized != query.strip():
        logger.info("Normalized query: %r -> %r", query, normalized)
    return normalized
