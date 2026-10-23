"""Single-agentic-rag's tool-calling loop - ported from hrb_chatbot_service's
ai/agents/workflow_agents/orchestration_agent.py. Uses ChatOpenAI directly via
build_agent_llm(), not a GatewayChatModel equivalent (none exists in this repo).

Simplified, flagged: live eval-judge scoring (evaluations/, DeepEval) is
deferred, same as Phase 1's genai/pipeline.py - eval_scores stays None on
every response, a graceful no-op not a crash. The MCP circuit breaker is
also not ported, same Phase 1 precedent (genai/mcp_tools.py already
documents this)."""

import asyncio
import hashlib
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from hrb_common.observability.call_logger import log_backend_call
from hrb_common.observability.logger import get_logger
from rag_retrieval.core.eval_threshold import build_eval_scores, retrieval_relevance_enabled, run_enabled_eval_judges
from rag_retrieval.core.guarded_pipeline import run_guarded_pipeline
from rag_retrieval.core.response_format import TABULAR_FORMAT_INSTRUCTION, should_use_tabular_format
from rag_retrieval.core.stores import AnswerCacheStore
from rag_retrieval.core.tool_classification import classify_tool
from rag_retrieval.genai.exact_knn_search import search_knn
from rag_retrieval.genai.mcp_tools import is_leave_balance_query, is_leave_history_query
from rag_retrieval.search_pre_processing import conversation_memory
from rag_retrieval.search_pre_processing.llm_helpers import AGENT_LLM_TIMEOUT_SECONDS, build_agent_llm
from rag_retrieval.search_pre_processing.prompts import ORCHESTRATION_SYSTEM_PROMPT
from rag_retrieval.search_pre_processing.query_normalize import normalize_query
from rag_retrieval.single_agentic.tools import (
    AGENTIC_TOOL_DEFINITIONS,
    get_leave_balance_tool,
    get_leave_history_tool,
    search_knowledge_base,
)

logger = get_logger("single_agentic.orchestration_agent")

DEFAULT_MAX_ITERATIONS = 5

# A cached answer can only be reused safely if no tool call in it could have
# returned live, per-employee data that's since changed - SearchKnowledgeBase
# queries the same static document index genai-rag's own retrieval does, so
# it's excluded from this set on purpose. Note: PascalCase here, matching
# this loop's own TOOL_FUNCTIONS keys - NOT the same casing as genai-rag's
# own search_pre_processing/cache_layer.py's LIVE_DATA_MCP_TOOLS (lowercase,
# MCP tool names), so that module's is_cache_eligible() isn't reused here -
# a real naming mismatch, not a copy-paste shortcut.
LIVE_DATA_TOOLS = {"GetLeaveBalance", "GetLeaveHistory"}

_answer_cache = AnswerCacheStore()


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def _message_role(message) -> str:
    kind = message.__class__.__name__
    return {"HumanMessage": "human", "AIMessage": "ai", "ToolMessage": "tool"}.get(kind, kind.lower())


def _render_turn(label: str, messages: list, response_text: str, latency_ms: float | None = None) -> dict:
    """One ReAct iteration's real, exact prompt: messages[0] is always the
    SystemMessage; the last of the rest is "what's new this turn" (the
    original query on iteration 1, a tool result after that)."""
    system_prompt = messages[0].content if messages else None
    rest = messages[1:]
    human_message = rest[-1].content if rest else None
    chat_history = [{"role": _message_role(message), "content": message.content} for message in rest[:-1]]
    return {
        "label": label,
        "latency_ms": latency_ms,
        "system_prompt": system_prompt,
        "human_message": human_message,
        "chat_history": chat_history,
        "response": response_text,
    }


# Same wider-pool approach genai/pipeline.py's own deferred judges would use
# (kept here for parity even though it's currently a no-op pending the
# evaluations/ deferral - see module docstring): context_texts (tool_outputs)
# mixes KB chunks with MCP text, which isn't something exact-KNN can compare
# against, so the retrieval judge scopes to sources (the structured, real
# KB-chunk list) instead of the mixed blob.
WIDER_POOL_MULTIPLIER = 4
WIDER_POOL_MAX = 20
# search_knowledge_base (single_agentic/tools.py) hardcodes top_k=3 for
# every call - mirrored here, not guessed.
SEARCH_KNOWLEDGE_BASE_TOP_K = 3


def _chunk_key(chunk: dict) -> tuple:
    return (chunk.get("document_id"), chunk.get("chunk_index"))


async def _fetch_wider_pool_texts(query: str, original_sources: list[dict]) -> list[str]:
    wider_k = min(SEARCH_KNOWLEDGE_BASE_TOP_K * WIDER_POOL_MULTIPLIER, WIDER_POOL_MAX)
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
    """Deferred this phase, same as genai/pipeline.py's own _score_live_answer -
    evaluations/ (DeepEval) isn't ported, so run_enabled_eval_judges() always
    returns (None, None, None, None) and eval_scores stays null, not a crash."""
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


SYSTEM_PROMPT = ORCHESTRATION_SYSTEM_PROMPT

TOOL_FUNCTIONS = {
    # SearchKnowledgeBase also takes the tool call's own search_strategy arg
    # (the agent's own choice) plus role/department (Document ACL - the
    # caller's own enforced identity, never the agent's choice).
    "SearchKnowledgeBase": lambda args, employee_id, role, department: search_knowledge_base(
        args.get("input", ""), args.get("search_strategy"), role=role, department=department
    ),
    "GetLeaveBalance": lambda args, employee_id, role, department: get_leave_balance_tool(employee_id),
    "GetLeaveHistory": lambda args, employee_id, role, department: get_leave_history_tool(employee_id),
}


def _build_llm() -> ChatOpenAI:
    return build_agent_llm()


def _build_cache_key(**kwargs) -> str:
    """Deterministic cache key from every option that can change the answer -
    same pattern genai/pipeline.py's own _build_cache_key() already uses."""
    canonical = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def run_agent(
    query: str,
    employee_id: str | None,
    max_iterations: int | None = None,
    enable_conversation_memory: bool = False,
    conversation_id: str | None = None,
    request_id: str | None = None,
    judge_model: str | None = None,
    eval_score_threshold: float | None = None,
    search_strategy: str | None = None,
    role: str | None = None,
    department: str | None = None,
    guardrails_model: str | None = None,
) -> dict:
    """Runs the tool-calling loop for one query - guardrailed and memory-
    tracked via the shared core (core/guarded_pipeline.py). Every result is
    saved to history, including the iteration-exhausted case - matches
    genai-rag's own behavior.

    search_strategy is a DEFAULT for SearchKnowledgeBase tool calls, used
    only when the agent's own tool call doesn't specify one - never forces
    the agent's own live choice. role/department (Document ACL) are the
    caller's own enforced identity, passed to every tool call uniformly."""
    started_at = time.perf_counter()
    normalized_query = await normalize_query(query)
    resolved_max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
    llm = _build_llm().bind(tools=AGENTIC_TOOL_DEFINITIONS)

    resolved_conversation_id = None
    if enable_conversation_memory:
        resolved_conversation_id = conversation_id or conversation_memory.new_conversation_id()

    # role/department included - same reasoning as genai/pipeline.py's own
    # cache key (a cached answer's citations can depend on which documents
    # the ORIGINAL caller's role/department could see).
    cache_key = _build_cache_key(
        query=normalized_query, pipeline="single-agentic-rag", max_iterations=resolved_max_iterations,
        search_strategy=search_strategy, role=role, department=department,
    )
    cached_result = _answer_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Answer cache hit for %r (single-agentic-rag)", normalized_query)
        result = dict(cached_result)
        result["conversation_id"] = resolved_conversation_id
        if resolved_conversation_id:
            await conversation_memory.save_turn(resolved_conversation_id, employee_id, normalized_query, result["answer"])
            _answer_cache.tag_conversation(resolved_conversation_id, cache_key)
        result["query"] = normalized_query
        result["served_from_cache"] = True
        result["llm_call_count"] = 0
        result["token_usage"] = None
        result["llm_context"] = None
        result["latency_ms"] = {"total": _elapsed_ms(started_at), "retrieval": None, "generation": None, "eval": None}
        return result

    async def generate(checked_query: str, chat_history: list | None) -> dict:
        system_prompt = SYSTEM_PROMPT
        if should_use_tabular_format(checked_query):
            system_prompt = f"{SYSTEM_PROMPT}\n\n{TABULAR_FORMAT_INSTRUCTION}"
        messages = [SystemMessage(content=system_prompt), *(chat_history or []), HumanMessage(content=checked_query)]
        tools_used = []
        tool_outputs = []  # raw tool output text, kept alongside tools_used - needed to eval groundedness
        sources = []  # structured RetrievedChunk-shaped dicts, for real citations
        turns = []  # one real LlmContextTurn per ReAct iteration
        total_prompt_tokens = 0
        total_completion_tokens = 0
        llm_call_count = 0
        generation_started_at = time.perf_counter()

        def _token_usage() -> dict | None:
            if llm_call_count == 0:
                return None
            return {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            }

        config = {"metadata": {"request_id": request_id}} if request_id else None
        for i in range(resolved_max_iterations):
            iteration_started_at = time.perf_counter()
            with log_backend_call(logger, "orchestration_agent", "ainvoke", iteration=i + 1, request_id=request_id):
                response = await asyncio.wait_for(llm.ainvoke(messages, config=config), timeout=AGENT_LLM_TIMEOUT_SECONDS)
            iteration_latency_ms = round((time.perf_counter() - iteration_started_at) * 1000, 1)
            llm_call_count += 1
            turns.append(_render_turn(f"iteration {i + 1}", messages, response.content, latency_ms=iteration_latency_ms))

            if response.usage_metadata:
                total_prompt_tokens += response.usage_metadata["input_tokens"]
                total_completion_tokens += response.usage_metadata["output_tokens"]
                logger.info(
                    "[orchestration_agent] tokens used: %s prompt + %s completion",
                    response.usage_metadata["input_tokens"],
                    response.usage_metadata["output_tokens"],
                )

            messages.append(response)

            if not response.tool_calls:
                return {
                    "answer": response.content,
                    "tools_used": tools_used,
                    "tool_outputs": tool_outputs,
                    "sources": sources,
                    "iterations": i + 1,
                    "llm_call_count": llm_call_count,
                    "token_usage": _token_usage(),
                    "llm_context": {"turns": turns, "mcp_tool_calls": []},
                    "latency_ms": {"total": None, "retrieval": None, "generation": _elapsed_ms(generation_started_at)},
                }

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_input = tool_args.get("input", "")

                # The caller's own search_strategy DEFAULT only fills in
                # when the agent's own tool call didn't already make its
                # own choice - never overrides it.
                if tool_name == "SearchKnowledgeBase" and "search_strategy" not in tool_args and search_strategy:
                    tool_args = {**tool_args, "search_strategy": search_strategy}

                tool_function = TOOL_FUNCTIONS.get(tool_name)
                tool_call_started_at = time.perf_counter()
                if tool_function is None:
                    tool_output, tool_chunks = f"Error: unknown tool {tool_name!r}", []
                else:
                    tool_output, tool_chunks = await tool_function(tool_args, employee_id, role, department)
                tool_call_latency_ms = _elapsed_ms(tool_call_started_at)

                tools_used.append(
                    {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_type": classify_tool(tool_name),
                        "search_strategy": tool_args.get("search_strategy"),
                        "latency_ms": tool_call_latency_ms,
                        "success": not tool_output.startswith("Error:"),
                        "iteration": i + 1,
                    }
                )
                tool_outputs.append(tool_output)
                sources.extend(tool_chunks)
                logger.info("Agent called tool=%s input=%r args=%r", tool_name, tool_input[:80], tool_args)
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return {
            "answer": "I wasn't able to finish reasoning about this within the allowed number of steps. Please rephrase or ask a more specific question.",
            "tools_used": tools_used,
            "tool_outputs": tool_outputs,
            "sources": sources,
            "iterations": resolved_max_iterations,
            "llm_call_count": llm_call_count,
            "token_usage": _token_usage(),
            "llm_context": {"turns": turns, "mcp_tool_calls": []},
            "latency_ms": {"total": None, "retrieval": None, "generation": _elapsed_ms(generation_started_at)},
        }

    result = await run_guarded_pipeline(
        normalized_query, employee_id, enable_conversation_memory, conversation_id, generate,
        guardrails_model=guardrails_model,
    )

    result["query"] = normalized_query
    result["served_from_cache"] = False
    eval_scores, eval_ms = await _score_live_answer(
        normalized_query, result["answer"], result.get("tool_outputs") or [],
        sources=result.get("sources") or [],
        judge_model=judge_model, eval_score_threshold=eval_score_threshold,
    )
    result["eval_scores"] = eval_scores
    result["latency_ms"]["eval"] = eval_ms
    result["latency_ms"]["total"] = _elapsed_ms(started_at)

    # A non-deterministic LLM tool choice must not get a free pass into a
    # 6h cache - if the query itself looks like it's asking for live
    # personal data, it stays uncached even when this one run happened not
    # to call the live-data tool.
    cache_eligible = (
        not any(call["tool_name"] in LIVE_DATA_TOOLS for call in result["tools_used"])
        and not is_leave_balance_query(normalized_query)
        and not is_leave_history_query(normalized_query)
    )
    if cache_eligible:
        _answer_cache.set(cache_key, normalized_query, result)
        if result.get("conversation_id"):
            _answer_cache.tag_conversation(result["conversation_id"], cache_key)

    return result
