"""Multi-agentic-rag's real LangGraph graph: Planner Agent -> Orchestration
dispatch (Send fan-out) -> 5 domain agents in parallel -> Reviewer Agent.
Hierarchical/LangGraph-supervisor pattern. Ported from hrb_chatbot_service's
ai/agents/workflow_agents/multi_agent_pipeline.py.

Simplified, flagged: live eval-judge scoring (evaluations/, DeepEval) is
deferred, same as Phase 1/2's own pipeline/orchestration_agent - eval_scores
stays None on every response, a graceful no-op not a crash. The MCP circuit
breaker is also not ported, same precedent."""

import asyncio
import hashlib
import json
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from hrb_common.observability.logger import get_logger
from rag_retrieval.core.eval_threshold import build_eval_scores, retrieval_relevance_enabled, run_enabled_eval_judges
from rag_retrieval.core.guarded_pipeline import run_guarded_pipeline
from rag_retrieval.core.stores import AnswerCacheStore
from rag_retrieval.core.tool_classification import classify_tool
from rag_retrieval.genai.exact_knn_search import search_knn
from rag_retrieval.genai.reviewer import generate_follow_ups, review
from rag_retrieval.multi_agentic import planner_agent
from rag_retrieval.multi_agentic.domain_agents import (
    lms_analytics_agent,
    lms_ops_agent,
    sql_db_agent,
    vector_kb_agent,
    web_search_agent,
)
from rag_retrieval.search_pre_processing import conversation_memory
from rag_retrieval.search_pre_processing.guardrails_input import check_input
from rag_retrieval.search_pre_processing.llm_helpers import resolve_agent_model
from rag_retrieval.search_pre_processing.query_normalize import normalize_query

logger = get_logger("multi_agentic.multi_agent_pipeline")

# Mirrors single_agentic/orchestration_agent.py's own LIVE_DATA_TOOLS, one
# level up at the agent-dispatch granularity - every domain agent except
# vector_kb_agent, which only queries the same static document index
# genai-rag's own retrieval does. Deliberately NOT built on top of
# search_pre_processing/cache_layer.py's own is_cache_eligible() - that
# checks lowercase MCP tool names, a different granularity than agent
# names, same real naming mismatch Phase 2's orchestration_agent.py
# already found and worked around once.
LIVE_DATA_AGENTS = {"lms_ops_agent", "sql_db_agent", "lms_analytics_agent", "web_search_agent"}

# vector_kb_agent.run() calls search_knowledge_base(focus) with no top_k
# override, which itself hardcodes top_k=3 (single_agentic/tools.py) -
# mirrored here, not guessed.
WIDER_POOL_MULTIPLIER = 4
WIDER_POOL_MAX = 20
VECTOR_KB_AGENT_TOP_K = 3

_answer_cache = AnswerCacheStore()


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def _build_cache_key(**kwargs) -> str:
    """Same pattern genai/pipeline.py's own _build_cache_key() already uses."""
    canonical = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _chunk_key(chunk: dict) -> tuple:
    return (chunk.get("document_id"), chunk.get("chunk_index"))


async def _fetch_wider_pool_texts(query: str, original_sources: list[dict]) -> list[str]:
    wider_k = min(VECTOR_KB_AGENT_TOP_K * WIDER_POOL_MULTIPLIER, WIDER_POOL_MAX)
    original_keys = {_chunk_key(chunk) for chunk in original_sources}
    extra_chunks = await asyncio.to_thread(search_knn, query, wider_k, None)
    deduped_extra = [chunk for chunk in extra_chunks if _chunk_key(chunk) not in original_keys]
    return [chunk["text"] for chunk in original_sources] + [chunk["text"] for chunk in deduped_extra]


async def _score_live_answer(
    query: str,
    answer: str,
    context_texts: list[str],
    sources: list[dict] | None = None,
    judge_model: str | None = None,
    eval_score_threshold: float | None = None,
) -> tuple[dict | None, float | None]:
    """Deferred this phase, same as Phase 1/2's own _score_live_answer -
    run_enabled_eval_judges() always returns (None, None, None, None), so
    eval_scores stays null, not a crash."""
    if not context_texts:
        return None, None

    kb_chunk_texts = [chunk["text"] for chunk in sources] if sources else []
    wider_pool_texts = (
        await _fetch_wider_pool_texts(query, sources) if sources and retrieval_relevance_enabled() else None
    )

    eval_started_at = time.perf_counter()
    groundedness, completeness, retrieval, answer_relevancy = await run_enabled_eval_judges(
        query, answer, context_texts, kb_chunk_texts, judge_model, wider_pool_texts
    )
    eval_scores = build_eval_scores(groundedness, completeness, retrieval, eval_score_threshold, answer_relevancy)
    return eval_scores, _elapsed_ms(eval_started_at)


DOMAIN_AGENT_NODES = [
    "vector_kb_agent",
    "lms_ops_agent",
    "sql_db_agent",
    "lms_analytics_agent",
    "web_search_agent",
]


class MultiAgentState(TypedDict):
    query: str
    employee_id: str | None
    request_id: str | None
    search_strategy: str | None
    role: str | None
    department: str | None
    planner_model: str | None
    tasks: list[dict]
    agent_results: Annotated[list[dict], operator.add]
    answer: str
    follow_up_questions: list[str]
    planner_turn: dict | None
    reviewer_turn: dict | None


async def planner_node(state: MultiAgentState) -> dict:
    tasks, turn = await planner_agent.plan(
        state["query"], request_id=state.get("request_id"), planner_model=state.get("planner_model")
    )
    return {"tasks": tasks, "planner_turn": turn}


def dispatch_to_agents(state: MultiAgentState) -> list[Send]:
    """The Orchestration Agent's dispatch step - fans each task out to its
    domain agent in parallel via LangGraph's own Send() API, not asyncio.gather()."""
    return [
        Send(
            task["agent"],
            {
                "query": state["query"],
                "employee_id": state["employee_id"],
                "focus": task["focus"],
                "search_strategy": state.get("search_strategy"),
                "role": state.get("role"),
                "department": state.get("department"),
            },
        )
        for task in state["tasks"]
    ]


async def vector_kb_agent_node(state: dict) -> dict:
    started_at = time.perf_counter()
    result, chunks = await vector_kb_agent.run(
        state["focus"], search_strategy=state.get("search_strategy"), role=state.get("role"), department=state.get("department")
    )
    return {
        "agent_results": [
            {"agent": "vector_kb_agent", "focus": state["focus"], "result": result, "sources": chunks, "latency_ms": _elapsed_ms(started_at)}
        ]
    }


async def lms_ops_agent_node(state: dict) -> dict:
    started_at = time.perf_counter()
    result, chunks = await lms_ops_agent.run(state["focus"], state["employee_id"])
    return {
        "agent_results": [
            {"agent": "lms_ops_agent", "focus": state["focus"], "result": result, "sources": chunks, "latency_ms": _elapsed_ms(started_at)}
        ]
    }


async def sql_db_agent_node(state: dict) -> dict:
    started_at = time.perf_counter()
    result = await sql_db_agent.run(state["focus"])
    return {
        "agent_results": [
            {"agent": "sql_db_agent", "focus": state["focus"], "result": result, "latency_ms": _elapsed_ms(started_at)}
        ]
    }


async def lms_analytics_agent_node(state: dict) -> dict:
    started_at = time.perf_counter()
    result = await lms_analytics_agent.run(state["focus"])
    return {
        "agent_results": [
            {"agent": "lms_analytics_agent", "focus": state["focus"], "result": result, "latency_ms": _elapsed_ms(started_at)}
        ]
    }


async def web_search_agent_node(state: dict) -> dict:
    started_at = time.perf_counter()
    result = await web_search_agent.run(state["focus"])
    return {
        "agent_results": [
            {"agent": "web_search_agent", "focus": state["focus"], "result": result, "latency_ms": _elapsed_ms(started_at)}
        ]
    }


async def reviewer_node(state: MultiAgentState) -> dict:
    answer, turn = await review(state["query"], state["agent_results"], request_id=state.get("request_id"))
    follow_up_questions = await generate_follow_ups(state["query"], answer) if state["agent_results"] else []
    return {"answer": answer, "follow_up_questions": follow_up_questions, "reviewer_turn": turn}


def _build_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("vector_kb_agent", vector_kb_agent_node)
    graph.add_node("lms_ops_agent", lms_ops_agent_node)
    graph.add_node("sql_db_agent", sql_db_agent_node)
    graph.add_node("lms_analytics_agent", lms_analytics_agent_node)
    graph.add_node("web_search_agent", web_search_agent_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch_to_agents, DOMAIN_AGENT_NODES)
    for node_name in DOMAIN_AGENT_NODES:
        graph.add_edge(node_name, "reviewer")
    graph.add_edge("reviewer", END)
    return graph.compile()


_GRAPH = None


def get_graph():
    """Builds the compiled graph once and reuses it - same lazy-singleton shape as this project's other gateways."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


async def run_multi_agent(
    query: str,
    employee_id: str | None,
    enable_conversation_memory: bool = False,
    conversation_id: str | None = None,
    request_id: str | None = None,
    judge_model: str | None = None,
    eval_score_threshold: float | None = None,
    search_strategy: str | None = None,
    role: str | None = None,
    department: str | None = None,
    planner_model: str | None = None,
    guardrails_model: str | None = None,
) -> dict:
    """Runs the Planner -> Orchestration -> domain agents -> Reviewer graph
    for one query. Guardrails run through the shared core (core/guarded_pipeline.py),
    same layer genai-rag/single-agentic-rag already use."""
    started_at = time.perf_counter()
    normalized_query = await normalize_query(query)
    checked_query, input_guardrail_status = await check_input(normalized_query, guardrails_model=guardrails_model)

    resolved_conversation_id = None
    if enable_conversation_memory:
        resolved_conversation_id = conversation_id or conversation_memory.new_conversation_id()

    cache_key = _build_cache_key(
        query=checked_query, pipeline="multi-agentic-rag", search_strategy=search_strategy, role=role, department=department
    )
    cached_result = _answer_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Answer cache hit for %r (multi-agentic-rag)", checked_query)
        result = dict(cached_result)
        result["conversation_id"] = resolved_conversation_id
        if resolved_conversation_id:
            await conversation_memory.save_turn(resolved_conversation_id, employee_id, checked_query, result["answer"])
            _answer_cache.tag_conversation(resolved_conversation_id, cache_key)
        result["served_from_cache"] = True
        result["llm_call_count"] = 0
        result["token_usage"] = None
        result["latency_ms"] = {"total": _elapsed_ms(started_at), "retrieval": None, "generation": None, "eval": None}
        result["llm_context"] = None
        result["planner_model_used"] = None
        result["query"] = checked_query
        return result

    async def generate(generate_query: str, chat_history: list | None) -> dict:
        graph = get_graph()
        graph_config = {"metadata": {"request_id": request_id}} if request_id else None
        final_state = await graph.ainvoke(
            {
                "query": generate_query,
                "employee_id": employee_id,
                "request_id": request_id,
                "search_strategy": search_strategy,
                "role": role,
                "department": department,
                "planner_model": planner_model,
                "tasks": [],
                "agent_results": [],
                "answer": "",
                "follow_up_questions": [],
                "planner_turn": None,
                "reviewer_turn": None,
            },
            config=graph_config,
        )

        agent_results = final_state["agent_results"]
        tools_used = [
            {
                "tool_name": agent_result["agent"],
                "tool_input": agent_result["focus"],
                "tool_type": classify_tool(agent_result["agent"]),
                "latency_ms": agent_result.get("latency_ms"),
                "success": not agent_result["result"].startswith("Error:"),
            }
            for agent_result in agent_results
        ]
        agent_result_texts = [agent_result["result"] for agent_result in agent_results]
        sources = [chunk for agent_result in agent_results for chunk in agent_result.get("sources", [])]

        turns = [turn for turn in (final_state.get("planner_turn"), final_state.get("reviewer_turn")) if turn]
        llm_context = {"turns": turns, "mcp_tool_calls": []} if turns else None

        return {
            "answer": final_state["answer"],
            "tasks": final_state["tasks"],
            "follow_up_questions": final_state["follow_up_questions"],
            "tools_used": tools_used,
            "agent_result_texts": agent_result_texts,
            "sources": sources,
            "iterations": len(agent_results),
            "served_from_cache": False,
            "llm_call_count": len(turns),
            "token_usage": None,
            "planner_model_used": resolve_agent_model("OPENAI_PLANNER_MODEL", planner_model),
            "llm_context": llm_context,
            "latency_ms": {"total": None, "retrieval": None, "generation": None, "eval": None},
        }

    result = await run_guarded_pipeline(
        checked_query, employee_id, enable_conversation_memory, conversation_id, generate,
        pre_checked_query=checked_query, pre_checked_input_status=input_guardrail_status,
        guardrails_model=guardrails_model,
    )
    result["query"] = checked_query

    eval_scores, eval_ms = await _score_live_answer(
        checked_query, result["answer"], result["agent_result_texts"], sources=result["sources"],
        judge_model=judge_model, eval_score_threshold=eval_score_threshold,
    )
    result["eval_scores"] = eval_scores
    result["latency_ms"]["eval"] = eval_ms
    result["latency_ms"]["total"] = _elapsed_ms(started_at)

    cache_eligible = not any(task["agent"] in LIVE_DATA_AGENTS for task in result["tasks"])
    if cache_eligible:
        _answer_cache.set(cache_key, checked_query, result)
        if result.get("conversation_id"):
            _answer_cache.tag_conversation(result["conversation_id"], cache_key)

    return result
