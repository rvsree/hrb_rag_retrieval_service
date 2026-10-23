"""Orchestrates answering one question against the indexed knowledge base -
ported from hrb_chatbot_service's ai/rag_pipeline/pipeline.py.

Simplified, flagged: live eval-judge scoring (_score_live_answer's own
wider-pool-fetch + evaluate_retrieval_relevance-based corrective retry) is
a no-op this phase - evaluations/ (DeepEval) is deferred (see
core/eval_threshold.py's own docstring). eval_scores will simply be None
on every response; enable_corrective_retry is accepted but never actually
retries (there's no live precision score yet to check against)."""

import time

from hrb_common.observability.logger import get_logger
from hrb_common.observability.secrets import read_secret
from rag_retrieval.core.eval_threshold import build_eval_scores, run_enabled_eval_judges
from rag_retrieval.core.guarded_pipeline import run_guarded_pipeline
from rag_retrieval.core.tool_classification import classify_tool
from rag_retrieval.genai.dynamic_strategy_selector import resolve_dynamic_strategy
from rag_retrieval.genai.mcp_tools import try_route_to_mcp
from rag_retrieval.genai.response_generator import generate_answer
from rag_retrieval.genai.retriever import retrieve_chunks
from rag_retrieval.genai.reviewer import review
from rag_retrieval.rag_query_params import RagQueryParams
from rag_retrieval.search_pre_processing import cache_layer, conversation_memory
from rag_retrieval.search_pre_processing.guardrails_input import check_input
from rag_retrieval.search_pre_processing.query_decompose import decompose
from rag_retrieval.search_pre_processing.query_normalize import normalize_query

logger = get_logger("genai.pipeline")


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def _build_cache_key(**kwargs) -> str:
    """Deterministic cache key from every option that can change the
    answer - same real intent as the original's build_cache_key(), a
    content-hash-free direct stable-repr (that project's own version hashes
    the query; this phase keeps it simple since the key is namespaced per-
    service anyway in AnswerCacheStore)."""
    import hashlib
    import json

    canonical = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _score_live_answer(
    query: str, answer: str, sources: list[dict], judge_model: str | None = None, eval_score_threshold: float | None = None,
) -> tuple[dict | None, float | None]:
    """DEFERRED this phase - evaluations/ (DeepEval) not ported, see module
    docstring. Always returns (None, None) when there's context to check
    (eval_scores all-None, not a real score), or (None, None) when there
    isn't - same return shape as the original either way, just never a
    real score this phase."""
    if not sources:
        return None, None
    eval_started_at = time.perf_counter()
    groundedness, completeness, retrieval, answer_relevancy = await run_enabled_eval_judges(
        query, answer, [c["text"] for c in sources], [c["text"] for c in sources], judge_model, None
    )
    eval_scores = build_eval_scores(groundedness, completeness, retrieval, eval_score_threshold, answer_relevancy)
    return eval_scores, _elapsed_ms(eval_started_at)


async def _answer_sub_question(
    sub_question: str,
    employee_id: str | None,
    resolved_top_k: int,
    resolved_vector_db: str,
    resolved_search_strategy: str,
    resolved_llm_provider: str,
    resolved_temperature: float,
    model_name: str | None,
    max_tokens: int | None,
    use_multi_query: bool,
    use_self_query: bool,
    chat_history: list,
    lambda_mult: float | None = None,
    request_id: str | None = None,
    role: str | None = None,
    department: str | None = None,
) -> dict:
    mcp_result = await try_route_to_mcp(sub_question, employee_id)
    if mcp_result is not None:
        return {
            "answer": mcp_result["answer"],
            "model_used": mcp_result["model_used"],
            "sources": [],
            "applied_filter": None,
            "tool_call": {
                "tool_name": mcp_result["routed_to"],
                "tool_input": sub_question,
                "tool_type": classify_tool(mcp_result["routed_to"]),
                "latency_ms": mcp_result.get("tool_latency_ms"),
                "success": True,
            },
            "token_usage": None,
            "llm_call_count": 0,
            "acl_denied_count": 0,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "llm_context_turn": None,
            "mcp_tool_call": {
                "tool_name": mcp_result["routed_to"],
                "arguments": mcp_result["mcp_arguments"],
                "raw_result": mcp_result["mcp_raw_result"],
            },
        }

    retrieval_started_at = time.perf_counter()
    chunks, applied_filter = await retrieve_chunks(
        sub_question, top_k=resolved_top_k, vector_db=resolved_vector_db, search_strategy=resolved_search_strategy,
        use_multi_query=use_multi_query, use_self_query=use_self_query, llm_provider=resolved_llm_provider,
        lambda_mult=lambda_mult, role=role, department=department,
    )
    from rag_retrieval.genai.access_control import build_acl_filter
    from rag_retrieval.genai.retriever import _metadata_store

    try:
        acl_filter_for_count = await build_acl_filter(_metadata_store, role, department)
    except Exception:
        acl_filter_for_count = None
    acl_denied_count = len(acl_filter_for_count["document_id"]["$nin"]) if acl_filter_for_count else 0
    retrieval_ms = _elapsed_ms(retrieval_started_at)

    generation_started_at = time.perf_counter()
    generation = generate_answer(
        sub_question, chunks, model_name=model_name, temperature=resolved_temperature,
        max_tokens=max_tokens, chat_history=chat_history, request_id=request_id,
    )
    generation_ms = _elapsed_ms(generation_started_at)

    return {
        "answer": generation["answer"],
        "model_used": generation["model_used"],
        "sources": chunks,
        "applied_filter": applied_filter,
        "acl_denied_count": acl_denied_count,
        "tool_call": None,
        "token_usage": generation.get("token_usage"),
        "llm_call_count": 1,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "llm_context_turn": generation.get("llm_context_turn"),
        "mcp_tool_call": None,
        "retry_attempted": False,
    }


async def answer_query(params: RagQueryParams, request_id: str | None = None) -> dict:
    """Run the full pipeline: guardrail/decompose/(MCP-route-or-retrieve+
    generate per sub-question)/merge/guardrail/save."""
    started_at = time.perf_counter()
    normalized_query = await normalize_query(params.query)
    checked_query, input_guardrail_status = await check_input(normalized_query, guardrails_model=params.guardrails_model)

    # These 3 go through read_secret() (override, then env, then default),
    # not a plain `or` literal - a bare `or` here silently skips this
    # service's real ACTIVE_VECTOR_DB/RAG_DEFAULT_SEARCH_STRATEGY/
    # ACTIVE_LLM_PROVIDER config and always falls back to the hardcoded
    # default, which is exactly the bug that was found live: production
    # sets ACTIVE_VECTOR_DB=pinecone, but every query silently used
    # chromadb (unwritable, empty) instead. Matches hrb_chatbot_service's
    # own get_active_vector_db()/get_active_llm_provider()/read_setting().
    resolved_vector_db = read_secret("ACTIVE_VECTOR_DB", "chromadb", passed_in_value=params.vector_db)
    resolved_search_strategy = read_secret(
        "RAG_DEFAULT_SEARCH_STRATEGY", "similarity", passed_in_value=params.search_strategy
    )
    if resolved_search_strategy == "dynamic":
        resolved_search_strategy = resolve_dynamic_strategy(checked_query)
    resolved_llm_provider = read_secret("ACTIVE_LLM_PROVIDER", "openai", passed_in_value=params.llm_provider)
    resolved_top_k = params.top_k or 5
    resolved_temperature = params.temperature if params.temperature is not None else 0.0

    cache_key = _build_cache_key(
        query=checked_query, top_k=resolved_top_k, vector_db=resolved_vector_db, search_strategy=resolved_search_strategy,
        model_name=params.model_name, temperature=resolved_temperature, max_tokens=params.max_tokens,
        use_multi_query=params.use_multi_query, use_self_query=params.use_self_query, llm_provider=resolved_llm_provider,
        lambda_mult=params.lambda_mult, role=params.role, department=params.department,
    )
    cache_hit = await cache_layer.check_cache(
        cache_key, checked_query, params.conversation_id, params.employee_id, params.eval_score_threshold, started_at
    )
    if cache_hit is not None:
        return cache_hit

    logger.info(
        "Answering query %r (top_k=%s, vector_db=%s, search_strategy=%s, llm_provider=%s)",
        checked_query, resolved_top_k, resolved_vector_db, resolved_search_strategy, resolved_llm_provider,
    )

    async def generate(generate_query: str, chat_history: list) -> dict:
        decomposition_started_at = time.perf_counter()
        sub_questions = await decompose(generate_query, decompose_model=params.decompose_model)
        decomposition_ms = _elapsed_ms(decomposition_started_at)

        sub_results = []
        for sub_question in sub_questions:
            sub_results.append(
                await _answer_sub_question(
                    sub_question, params.employee_id, resolved_top_k, resolved_vector_db, resolved_search_strategy,
                    resolved_llm_provider, resolved_temperature, params.model_name, params.max_tokens,
                    params.use_multi_query, params.use_self_query, chat_history,
                    lambda_mult=params.lambda_mult, request_id=request_id, role=params.role, department=params.department,
                )
            )
        retrieval_ms = sum(r["retrieval_ms"] for r in sub_results)
        generation_ms = sum(r["generation_ms"] for r in sub_results)

        sources = [chunk for r in sub_results for chunk in r["sources"]]
        applied_filter = next((r["applied_filter"] for r in sub_results if r["applied_filter"]), None)
        acl_denied_count = max((r.get("acl_denied_count", 0) for r in sub_results), default=0)
        tools_used = [r["tool_call"] for r in sub_results if r["tool_call"] is not None]
        llm_call_count = 1 + sum(r["llm_call_count"] for r in sub_results)

        llm_context_turns = []
        mcp_tool_calls = []
        for sub_question, r in zip(sub_questions, sub_results, strict=True):
            if r["llm_context_turn"] is not None:
                turn = dict(r["llm_context_turn"])
                if len(sub_results) > 1:
                    turn["label"] = sub_question
                llm_context_turns.append(turn)
            if r["mcp_tool_call"] is not None:
                mcp_tool_calls.append(r["mcp_tool_call"])

        if len(sub_results) == 1:
            answer = sub_results[0]["answer"]
            model_used = sub_results[0]["model_used"]
            token_usage = sub_results[0]["token_usage"]
        else:
            logger.info("Query %r decomposed into %d sub-questions", generate_query, len(sub_results))
            agent_results = [
                {"agent": "genai-rag", "focus": sub_question, "result": r["answer"]}
                for sub_question, r in zip(sub_questions, sub_results, strict=True)
            ]
            answer, merge_turn = await review(generate_query, agent_results, request_id=request_id)
            llm_call_count += 1
            llm_context_turns.append(
                {
                    "label": f"Merging {len(sub_results)} sub-answers into one response",
                    "system_prompt": merge_turn["system_prompt"] if merge_turn else None,
                    "human_message": merge_turn["human_message"] if merge_turn else None,
                    "chat_history": [],
                    "response": answer,
                }
            )
            models_used = {r["model_used"] for r in sub_results}
            model_used = ", ".join(sorted(models_used))
            prompt_tokens = sum(r["token_usage"]["prompt_tokens"] for r in sub_results if r["token_usage"])
            completion_tokens = sum(r["token_usage"]["completion_tokens"] for r in sub_results if r["token_usage"])
            token_usage = (
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
                if any(r["token_usage"] for r in sub_results)
                else None
            )

        logger.info("Query %r answered using %d chunk(s) across %d sub-question(s)", generate_query, len(sources), len(sub_results))
        return {
            "query": generate_query,
            "answer": answer,
            "model_used": model_used,
            "sources": sources,
            "vector_db": resolved_vector_db,
            "search_strategy": resolved_search_strategy,
            "applied_filter": applied_filter,
            "acl_denied_count": acl_denied_count,
            "served_from_cache": False,
            "llm_call_count": llm_call_count,
            "token_usage": token_usage,
            "temperature": resolved_temperature,
            "decompose_model_used": params.decompose_model or "gpt-4.1-nano",
            "tools_used": tools_used,
            "retry_attempted": False,
            "llm_context": {"turns": llm_context_turns, "mcp_tool_calls": mcp_tool_calls},
            "latency_ms": {"total": None, "retrieval": retrieval_ms, "generation": generation_ms, "decomposition": decomposition_ms},
        }

    result = await run_guarded_pipeline(
        checked_query, params.employee_id, params.enable_conversation_memory, params.conversation_id, generate,
        pre_checked_query=checked_query, pre_checked_input_status=input_guardrail_status, guardrails_model=params.guardrails_model,
    )
    result["eval_scores"], result["latency_ms"]["eval"] = await _score_live_answer(
        checked_query, result["answer"], result["sources"], judge_model=params.judge_model, eval_score_threshold=params.eval_score_threshold,
    )
    if result["eval_scores"] is not None:
        result["eval_scores"]["retrieval_retry_attempted"] = result.get("retry_attempted", False)

    result["latency_ms"]["total"] = _elapsed_ms(started_at)
    await cache_layer.write_cache(cache_key, checked_query, result)
    return result
