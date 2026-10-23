"""Single-agentic-rag's own query route - a tool-calling agent, not a fixed
retrieve-then-generate pipeline. Ported from hrb_chatbot_service's
api/agentic_rag/query_agent.py.

Simplified, flagged: no RBAC, no rate limiting, no request_metrics logging -
same "accepts requests with no identity check for now" precedent this
repo's own routes.py (Phase 1) already set. request_id is generated
directly via uuid4() instead of going through the original's
metrics-storage wrapper (time_and_log)."""

import uuid

from fastapi import APIRouter, HTTPException

from rag_retrieval.models import (
    AgenticRagRequest,
    AgenticRagResponse,
    EvalScores,
    ExplainabilityInfo,
    LatencyInfo,
    LlmContextInfo,
    TokenUsageInfo,
    ToolCallInfo,
)
from rag_retrieval.search_pre_processing.guardrails_input import GuardrailBlockedError
from rag_retrieval.single_agentic.orchestration_agent import run_agent

router = APIRouter(tags=["single-agentic-rag"])


@router.post("/single-agentic-rag/query", response_model=AgenticRagResponse)
async def query(payload: AgenticRagRequest):
    employee_id = payload.user_profile.employee_id if payload.user_profile else None
    role = payload.user_profile.role if payload.user_profile else None
    department = payload.user_profile.department if payload.user_profile else None
    request_id = str(uuid.uuid4())

    try:
        result = await run_agent(
            payload.query,
            employee_id,
            payload.max_iterations,
            enable_conversation_memory=payload.enable_conversation_memory,
            conversation_id=payload.conversation_id,
            request_id=request_id,
            judge_model=payload.eval_options.judge_model if payload.eval_options else None,
            eval_score_threshold=payload.eval_options.eval_score_threshold if payload.eval_options else None,
            search_strategy=payload.search_strategy,
            role=role,
            department=department,
            guardrails_model=payload.guardrails_model,
        )
    except GuardrailBlockedError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    token_usage = result.get("token_usage")
    eval_scores = result.get("eval_scores")
    llm_context = result.get("llm_context")
    return AgenticRagResponse(
        query=result.get("query", payload.query),
        answer=result["answer"],
        tools_used=[ToolCallInfo(**call) for call in result["tools_used"]],
        sources=result.get("sources") or [],
        iterations=result["iterations"],
        explainability_info=ExplainabilityInfo(
            request_id=request_id,
            served_from_cache=result["served_from_cache"],
            llm_call_count=result["llm_call_count"],
            latency_ms=LatencyInfo(**result["latency_ms"]),
            token_usage=TokenUsageInfo(**token_usage) if token_usage else None,
            llm_context=LlmContextInfo(**llm_context) if llm_context else None,
            routed_to=None,
            eval_scores=EvalScores(**eval_scores) if eval_scores else None,
            guardrails=result["guardrails"],
            guardrails_model_used=result.get("guardrails_model_used"),
        ),
        conversation_id=result.get("conversation_id"),
    )
