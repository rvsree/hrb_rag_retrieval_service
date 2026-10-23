"""The guardrail/conversation-memory/logging wrapper every RAG retrieval
mode shares - ported from hrb_chatbot_service's ai/rag_core/guarded_pipeline.py."""

from collections.abc import Awaitable, Callable

from langchain_core.messages import BaseMessage

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.genai.guardrails_output import check_output
from rag_retrieval.search_pre_processing import conversation_memory
from rag_retrieval.search_pre_processing.guardrails_input import check_input

logger = get_logger("core.guarded_pipeline")

GenerateFn = Callable[[str, list[BaseMessage] | None], Awaitable[dict]]


async def run_guarded_pipeline(
    query: str,
    employee_id: str | None,
    enable_conversation_memory: bool,
    conversation_id: str | None,
    generate: GenerateFn,
    pre_checked_query: str | None = None,
    pre_checked_input_status: dict | None = None,
    guardrails_model: str | None = None,
) -> dict:
    """Input guardrail -> load history -> generate (mode-specific) -> output
    guardrail -> save turn."""
    if pre_checked_query is not None:
        checked_query = pre_checked_query
        input_status = pre_checked_input_status or {"status": "pass", "rail": None}
    else:
        checked_query, input_status = await check_input(query, guardrails_model=guardrails_model)

    resolved_conversation_id = None
    if enable_conversation_memory:
        resolved_conversation_id = conversation_id or conversation_memory.new_conversation_id()
    chat_history = await conversation_memory.load_history(resolved_conversation_id) if resolved_conversation_id else None

    with log_backend_call(logger, "rag_core", "generate"):
        result = await generate(checked_query, chat_history)

    checked_answer, output_status = await check_output(checked_query, result["answer"], guardrails_model=guardrails_model)

    if resolved_conversation_id:
        await conversation_memory.save_turn(resolved_conversation_id, employee_id, checked_query, checked_answer)

    result["answer"] = checked_answer
    result["conversation_id"] = resolved_conversation_id
    result["guardrails"] = {"input": input_status, "output": output_status}
    result["guardrails_model_used"] = output_status.get("model_used")
    return result
