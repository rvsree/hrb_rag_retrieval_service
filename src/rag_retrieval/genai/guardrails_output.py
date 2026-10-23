"""Checks a generated answer against NeMo Guardrails' output rails - ported
from hrb_chatbot_service's ai/rag_pipeline/response_generation/guardrails_output/__init__.py."""

from hrb_common.observability.logger import get_logger
from rag_retrieval.search_pre_processing.guardrails_input import (
    activated_rails_to_dicts,
    get_rails,
    resolve_guardrails_model,
)
from rag_retrieval.search_pre_processing.safe_terms import protect_known_safe_terms

logger = get_logger("guardrails_output")

SAFE_FALLBACK_ANSWER = "I'm not able to share that response. Please rephrase your question."


async def check_output(query: str, answer: str, guardrails_model: str | None = None) -> tuple[str, dict]:
    resolved_model = resolve_guardrails_model(guardrails_model)
    protected_answer = protect_known_safe_terms(answer)
    messages = [{"role": "user", "content": query}, {"role": "assistant", "content": protected_answer}]
    response = await get_rails(resolved_model).generate_async(
        messages=messages, options={"rails": ["output"], "log": {"activated_rails": True}}
    )
    activated_rails = response.log.activated_rails if response.log else []
    blocking_rail = next((rail.name for rail in activated_rails if rail.stop), None)
    result_content = response.response[-1]["content"] if isinstance(response.response, list) else response.response

    if blocking_rail:
        logger.warning("Output blocked by guardrail %s", blocking_rail)
        return SAFE_FALLBACK_ANSWER, {
            "status": "blocked",
            "rail": blocking_rail,
            "activated_rails": activated_rails_to_dicts(activated_rails),
            "model_used": resolved_model,
        }

    if result_content != protected_answer:
        logger.info("Output modified by guardrail - sensitive data masked")
        return result_content, {
            "status": "modified",
            "rail": None,
            "activated_rails": activated_rails_to_dicts(activated_rails),
            "model_used": resolved_model,
        }

    return result_content, {
        "status": "pass",
        "rail": None,
        "activated_rails": activated_rails_to_dicts(activated_rails),
        "model_used": resolved_model,
    }
