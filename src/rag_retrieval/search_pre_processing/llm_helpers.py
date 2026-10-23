"""Shared ChatOpenAI builder for query_decompose.py/query_normalize.py/
mode_router.py's own structured-output LLM calls - ported from
hrb_chatbot_service's ai/agents/_llm_helpers.py, same resolution order
(override > env setting > default)."""

from langchain_openai import ChatOpenAI

from hrb_common.observability.secrets import read_secret

AGENT_LLM_TIMEOUT_SECONDS = 30


def resolve_agent_model(model_setting_name: str | None = None, model_override: str | None = None) -> str:
    default_model = read_secret("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    env_model = read_secret(model_setting_name, default_model) if model_setting_name else default_model
    return model_override or env_model


def build_agent_llm(model_setting_name: str | None = None, model_override: str | None = None) -> ChatOpenAI:
    api_key = read_secret("OPENAI_API_KEY", "")
    base_url = (read_secret("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/")
    model = resolve_agent_model(model_setting_name, model_override)
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
