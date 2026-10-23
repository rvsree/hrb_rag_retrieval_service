"""The auto-router's single-vs-multi classifier - ported from
hrb_chatbot_service's ai/search_pre_processing/mode_router.py. Not wired
into genai-rag's own route this phase (used by the separate auto-router
endpoint, out of this phase's scope) - ported for completeness per the
spec's own "port ai/search_pre_processing/ in full" instruction."""

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.search_pre_processing.llm_helpers import AGENT_LLM_TIMEOUT_SECONDS, build_agent_llm
from rag_retrieval.search_pre_processing.prompts import ROUTER_SYSTEM_PROMPT

logger = get_logger("search_pre_processing.mode_router")


class ModeClassification(BaseModel):
    mode: Literal["single", "multi"]


def _build_llm(router_model: str | None = None):
    return build_agent_llm("OPENAI_ROUTER_MODEL", model_override=router_model)


async def classify_query_complexity(query: str, router_model: str | None = None) -> Literal["single", "multi"]:
    """Never raises - falls back to "single" on any classifier failure."""
    try:
        llm = _build_llm(router_model).with_structured_output(ModeClassification, include_raw=True)
        messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=query)]
        with log_backend_call(logger, "mode_router", "classify_query_complexity"):
            result = await asyncio.wait_for(llm.ainvoke(messages), timeout=AGENT_LLM_TIMEOUT_SECONDS)
    except Exception as error:
        logger.warning("Mode classification failed for %r, defaulting to single: %s: %s", query, type(error).__name__, error)
        return "single"

    output: ModeClassification | None = result["parsed"]
    if output is None:
        logger.warning("Mode classification didn't parse for %r, defaulting to single", query)
        return "single"
    return output.mode
