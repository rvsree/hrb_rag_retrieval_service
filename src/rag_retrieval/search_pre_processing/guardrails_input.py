"""Checks a query against NeMo Guardrails' input rails before retrieval -
ported from hrb_chatbot_service's ai/search_pre_processing/guardrails_input.py,
same logic, CONFIG_PATH adjusted to this repo's own guardrails_config/."""

import copy
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig

from hrb_common.observability.logger import get_logger
from hrb_common.observability.secrets import read_secret
from rag_retrieval.search_pre_processing.safe_terms import protect_known_safe_terms

logger = get_logger("guardrails_input")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "guardrails_config"

# 2026-10-23 HOTFIX: was "gpt-4.1-nano" - OpenAI shut that model down today
# (GET /v1/models/gpt-4.1-nano returns shutdown_date 2026-10-23; confirmed
# live, same real incident hrb_chatbot_service's own guardrails_input.py
# hit). gpt-4.1-mini's own shutdown_date is 2027-01-14.
DEFAULT_GUARDRAILS_MODEL = "gpt-4.1-mini"


def supported_guardrails_models() -> list[str]:
    raw = read_secret("GUARDRAILS_SUPPORTED_MODELS", f"{DEFAULT_GUARDRAILS_MODEL},gpt-4.1")
    return [model.strip() for model in raw.split(",") if model.strip()]


def resolve_guardrails_model(override: str | None = None) -> str:
    if override is None:
        return DEFAULT_GUARDRAILS_MODEL
    supported = supported_guardrails_models()
    if override not in supported:
        raise ValueError(f"guardrails_model {override!r} is not supported - choose one of {supported}.")
    return override


_base_config: RailsConfig | None = None
_rails_by_model: dict[str, LLMRails] = {}


def get_rails(model: str | None = None) -> LLMRails:
    global _base_config
    resolved_model = model or DEFAULT_GUARDRAILS_MODEL
    if resolved_model not in _rails_by_model:
        if _base_config is None:
            _base_config = RailsConfig.from_path(str(CONFIG_PATH))
        config = _base_config
        if resolved_model != _base_config.models[0].model:
            config = copy.deepcopy(_base_config)
            config.models[0].model = resolved_model
        _rails_by_model[resolved_model] = LLMRails(config)
    return _rails_by_model[resolved_model]


class GuardrailBlockedError(Exception):
    """Raised when a guardrail blocks a query - the route turns this into a 422."""


def activated_rails_to_dicts(activated_rails) -> list[dict]:
    return [{"name": rail.name, "decisions": list(rail.decisions), "stop": rail.stop} for rail in activated_rails]


async def check_input(query: str, guardrails_model: str | None = None) -> tuple[str, dict]:
    resolved_model = resolve_guardrails_model(guardrails_model)
    protected_query = protect_known_safe_terms(query)
    response = await get_rails(resolved_model).generate_async(
        messages=[{"role": "user", "content": protected_query}],
        options={"rails": ["input"], "log": {"activated_rails": True}},
    )
    activated_rails = response.log.activated_rails if response.log else []
    blocking_rail = next((rail.name for rail in activated_rails if rail.stop), None)
    result_content = response.response[-1]["content"] if isinstance(response.response, list) else response.response

    if blocking_rail:
        logger.warning("Input blocked by guardrail %s", blocking_rail)
        raise GuardrailBlockedError(f"This request was blocked by a safety check ({blocking_rail}).")

    if result_content != protected_query:
        logger.info("Input modified by guardrail - sensitive data masked")
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
