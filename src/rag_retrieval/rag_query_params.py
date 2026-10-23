"""Plain, framework-free container for one /query request - ported as-is from
hrb_chatbot_service's common/rag_query_params.py."""

from dataclasses import dataclass


@dataclass
class RagQueryParams:
    query: str
    top_k: int | None = None
    vector_db: str | None = None
    search_strategy: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    use_multi_query: bool = False
    use_self_query: bool = False
    llm_provider: str | None = None
    lambda_mult: float | None = None
    employee_id: str | None = None
    role: str | None = None
    department: str | None = None
    enable_conversation_memory: bool = False
    conversation_id: str | None = None
    judge_model: str | None = None
    eval_score_threshold: float | None = None
    enable_corrective_retry: bool = False
    decompose_model: str | None = None
    guardrails_model: str | None = None
