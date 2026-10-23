"""Ask the knowledge base a question - ported from hrb_chatbot_service's
api/rag/retrieve_document.py.

Simplified, flagged: no RBAC (require_role), no rate limiting
(enforce_rate_limit), no request_metrics logging (time_and_log) - this
repo has none of those layers yet, same "accepts requests with no identity
check for now" precedent rag_ingestion_service's own Phase 1 already set.
request_id is generated directly via uuid4() instead of going through the
original's metrics-storage wrapper."""

import uuid

from fastapi import APIRouter, HTTPException

from rag_retrieval.genai.pipeline import answer_query
from rag_retrieval.models import (
    AnswerInfo,
    EvalScores,
    ExplainabilityInfo,
    LatencyInfo,
    LlmContextInfo,
    RagQueryRequest,
    RagQueryResponse,
    RetrievalInfo,
    ToolCallInfo,
    TokenUsageInfo,
)
from rag_retrieval.rag_query_params import RagQueryParams
from rag_retrieval.search_pre_processing.guardrails_input import GuardrailBlockedError

router = APIRouter(tags=["query"])


def _params_from_request(payload: RagQueryRequest, employee_id: str | None = None, role: str | None = None, department: str | None = None) -> RagQueryParams:
    search_options = payload.search_options
    generation_options = payload.generation_options
    eval_options = payload.eval_options
    classification_options = payload.classification_options
    return RagQueryParams(
        query=payload.query,
        top_k=search_options.top_k if search_options else None,
        vector_db=search_options.vector_db if search_options else None,
        search_strategy=search_options.search_strategy if search_options else None,
        model_name=generation_options.model_name if generation_options else None,
        temperature=generation_options.temperature if generation_options else None,
        max_tokens=generation_options.max_tokens if generation_options else None,
        use_multi_query=search_options.use_multi_query if search_options else False,
        use_self_query=search_options.use_self_query if search_options else False,
        llm_provider=search_options.llm_provider if search_options else None,
        lambda_mult=search_options.lambda_mult if search_options else None,
        enable_corrective_retry=search_options.enable_corrective_retry if search_options else False,
        employee_id=employee_id,
        role=role,
        department=department,
        enable_conversation_memory=payload.enable_conversation_memory,
        conversation_id=payload.conversation_id,
        judge_model=eval_options.judge_model if eval_options else None,
        eval_score_threshold=eval_options.eval_score_threshold if eval_options else None,
        decompose_model=classification_options.decompose_model if classification_options else None,
        guardrails_model=payload.guardrails_model,
    )


async def _answer_query(payload: RagQueryRequest, employee_id: str | None = None, role: str | None = None, department: str | None = None) -> RagQueryResponse:
    params = _params_from_request(payload, employee_id, role, department)
    request_id = str(uuid.uuid4())
    try:
        result = await answer_query(params, request_id=request_id)
    except GuardrailBlockedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    token_usage = result.get("token_usage")
    eval_scores = result.get("eval_scores")
    return RagQueryResponse(
        query=result["query"],
        answer_info=AnswerInfo(answer=result["answer"], model_used=result["model_used"]),
        retrieval_info=RetrievalInfo(
            vector_db=result["vector_db"],
            search_strategy=result["search_strategy"],
            applied_filter=result.get("applied_filter"),
            acl_denied_count=result.get("acl_denied_count", 0),
            sources=result["sources"],
        ),
        explainability_info=ExplainabilityInfo(
            request_id=result.get("request_id", request_id),
            served_from_cache=result["served_from_cache"],
            llm_call_count=result["llm_call_count"],
            latency_ms=LatencyInfo(**result["latency_ms"]),
            token_usage=TokenUsageInfo(**token_usage) if token_usage else None,
            routed_to=result.get("routed_to"),
            eval_scores=EvalScores(**eval_scores) if eval_scores else None,
            temperature=result.get("temperature"),
            llm_context=LlmContextInfo(**result["llm_context"]) if result.get("llm_context") else None,
            guardrails=result["guardrails"],
            decompose_model_used=result.get("decompose_model_used"),
            guardrails_model_used=result.get("guardrails_model_used"),
        ),
        tools_used=[ToolCallInfo(**call) for call in result.get("tools_used", [])],
        conversation_id=result.get("conversation_id"),
    )


@router.post("/query", response_model=RagQueryResponse)
async def query(payload: RagQueryRequest):
    return await _answer_query(
        payload,
        employee_id=payload.user_profile.employee_id if payload.user_profile else None,
        role=payload.user_profile.role if payload.user_profile else None,
        department=payload.user_profile.department if payload.user_profile else None,
    )
