"""CAG (answer cache) check/write - ported from hrb_chatbot_service's
ai/search_pre_processing/cache_layer.py, adapted to this repo's own
core/stores.py AnswerCacheStore (Redis-backed, the real current
implementation - the old Postgres answer_cache table is orphaned)."""

import time

from hrb_common.observability.logger import get_logger
from rag_retrieval.core.eval_threshold import check_eval_threshold
from rag_retrieval.core.stores import AnswerCacheStore
from rag_retrieval.search_pre_processing import conversation_memory

logger = get_logger("search_pre_processing.cache_layer")

LIVE_DATA_MCP_TOOLS = {"get_leave_balance", "get_leave_history"}

_answer_cache = AnswerCacheStore()


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


async def check_cache(
    cache_key: str,
    checked_query: str,
    resolved_conversation_id: str | None,
    employee_id: str,
    eval_score_threshold: float | None,
    started_at: float,
) -> dict | None:
    """Returns the shaped hit result, or None on a miss."""
    cached_result = _answer_cache.get(cache_key)
    if cached_result is None:
        return None

    logger.info("Answer cache hit for %r", checked_query)
    result = dict(cached_result)
    result["conversation_id"] = resolved_conversation_id
    if resolved_conversation_id:
        await conversation_memory.save_turn(resolved_conversation_id, employee_id, checked_query, result["answer"])
        _answer_cache.tag_conversation(resolved_conversation_id, cache_key)

    if result.get("eval_scores") is not None:
        result["eval_scores"] = dict(result["eval_scores"])
        result["eval_scores"]["below_threshold"], result["eval_scores"]["below_threshold_metrics"] = (
            check_eval_threshold(result["eval_scores"], eval_score_threshold)
        )
    result["served_from_cache"] = True
    result["llm_call_count"] = 0
    result["token_usage"] = None
    result["latency_ms"] = {"total": _elapsed_ms(started_at), "retrieval": None, "generation": None, "eval": None}
    result["llm_context"] = None
    result["decompose_model_used"] = None
    return result


def is_cache_eligible(tools_used: list[dict]) -> bool:
    return not any(call["tool_name"] in LIVE_DATA_MCP_TOOLS for call in tools_used)


async def write_cache(cache_key: str, checked_query: str, result: dict) -> None:
    if not is_cache_eligible(result["tools_used"]):
        return
    _answer_cache.set(cache_key, checked_query, result)
    if result.get("conversation_id"):
        _answer_cache.tag_conversation(result["conversation_id"], cache_key)
