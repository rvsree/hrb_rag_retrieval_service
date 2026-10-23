"""genai-rag's own query-decomposition step - ported from hrb_chatbot_service's
ai/search_pre_processing/query_decompose.py, same structured-output pattern."""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.search_pre_processing.llm_helpers import AGENT_LLM_TIMEOUT_SECONDS, build_agent_llm
from rag_retrieval.search_pre_processing.prompts import DECOMPOSE_SYSTEM_PROMPT

logger = get_logger("pre_processing.query_decompose")


class DecomposedQuery(BaseModel):
    sub_questions: list[str]


def _build_llm(decompose_model: str | None = None) -> ChatOpenAI:
    return build_agent_llm("OPENAI_DECOMPOSE_MODEL", model_override=decompose_model)


async def decompose(query: str, decompose_model: str | None = None) -> list[str]:
    """Never returns an empty list - falls back to [query] unchanged on a
    parse failure or an empty result."""
    llm = _build_llm(decompose_model).with_structured_output(DecomposedQuery, include_raw=True)
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT), HumanMessage(content=query)]

    try:
        with log_backend_call(logger, "query_decompose", "decompose"):
            result = await asyncio.wait_for(llm.ainvoke(messages), timeout=AGENT_LLM_TIMEOUT_SECONDS)
    except Exception as error:
        logger.warning("Decomposition failed for %r, answering as one question: %s: %s", query, type(error).__name__, error)
        return [query]

    output: DecomposedQuery | None = result["parsed"]
    sub_questions = [q.strip() for q in output.sub_questions if q.strip()] if output else []
    if not sub_questions:
        return [query]

    if len(sub_questions) > 1:
        logger.info("Decomposed %r into %d sub-questions", query, len(sub_questions))
    return sub_questions
