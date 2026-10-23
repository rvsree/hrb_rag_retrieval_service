"""Tests for the ReAct tool-calling loop - the LLM itself is faked (no real
API call), matching every other client test in this project. Each fake
mirrors just enough of ChatOpenAI's shape (.bind().ainvoke()) to drive the
loop's branching. Judges are already deferred at the core.eval_threshold
level (Phase 1's own design - always returns (None, None, None, None)), so
no judge faking needed here, only the wider-pool search_knn call when
sources is non-empty."""

from langchain_core.messages import AIMessage

from rag_retrieval.core import guarded_pipeline
from rag_retrieval.search_pre_processing import conversation_memory
from rag_retrieval.single_agentic import orchestration_agent


class FakeAnswerCache:
    def __init__(self):
        self.store: dict[str, dict] = {}
        self.tagged: list[tuple[str, str]] = []

    def get(self, cache_key):
        return self.store.get(cache_key)

    def set(self, cache_key, checked_query, result):
        self.store[cache_key] = result

    def tag_conversation(self, conversation_id, cache_key):
        self.tagged.append((conversation_id, cache_key))


class FakeBoundLLM:
    """Returns one scripted AIMessage per call, in order."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls: list[list] = []

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        return self.responses.pop(0)


def _ai_message(content: str, tool_calls=None) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _patch_common(monkeypatch, bound_llm: FakeBoundLLM, answer_cache: FakeAnswerCache | None = None):
    async def _passthrough_check_input(query, guardrails_model=None):
        return query, {"status": "pass", "rail": None}

    async def _passthrough_check_output(query, answer, guardrails_model=None):
        return answer, {"status": "pass", "rail": None, "model_used": "gpt-4.1-nano"}

    async def _fake_load_history(conversation_id):
        return []

    async def _fake_save_turn(conversation_id, employee_id, human_content, ai_content):
        return None

    async def _passthrough_normalize(query):
        return query

    class _FakeUnboundLLM:
        def bind(self, tools):
            return bound_llm

    monkeypatch.setattr(guarded_pipeline, "check_input", _passthrough_check_input)
    monkeypatch.setattr(guarded_pipeline, "check_output", _passthrough_check_output)
    monkeypatch.setattr(conversation_memory, "load_history", _fake_load_history)
    monkeypatch.setattr(conversation_memory, "save_turn", _fake_save_turn)
    monkeypatch.setattr(orchestration_agent, "normalize_query", _passthrough_normalize)
    monkeypatch.setattr(orchestration_agent, "search_knn", lambda query, top_k, vector_db: [])
    monkeypatch.setattr(orchestration_agent, "_build_llm", lambda: _FakeUnboundLLM())
    monkeypatch.setattr(orchestration_agent, "_answer_cache", answer_cache or FakeAnswerCache())


async def test_no_tool_call_answers_directly(monkeypatch):
    bound_llm = FakeBoundLLM([_ai_message("Direct answer, no tool needed.")])
    _patch_common(monkeypatch, bound_llm)

    result = await orchestration_agent.run_agent("what is the company name", "EMP1")

    assert result["answer"] == "Direct answer, no tool needed."
    assert result["iterations"] == 1
    assert result["tools_used"] == []
    assert result["served_from_cache"] is False


async def test_one_tool_call_then_final_answer(monkeypatch):
    async def _fake_search_knowledge_base(query, search_strategy, role=None, department=None):
        return "Found relevant knowledge base content:\n\n--- Source 1 (policy.pdf) ---\nPTO details.\n\n", [
            {"document_id": "doc1", "chunk_index": 0, "filename": "policy.pdf", "text": "PTO details.", "score": 0.9}
        ]

    monkeypatch.setattr(orchestration_agent, "search_knowledge_base", _fake_search_knowledge_base)

    tool_call_message = _ai_message(
        "",
        tool_calls=[{"name": "SearchKnowledgeBase", "args": {"input": "PTO policy"}, "id": "call_1"}],
    )
    final_message = _ai_message("Per policy.pdf, you get 20 PTO days.")
    bound_llm = FakeBoundLLM([tool_call_message, final_message])
    _patch_common(monkeypatch, bound_llm)

    result = await orchestration_agent.run_agent("what is the PTO policy", "EMP1")

    assert result["answer"] == "Per policy.pdf, you get 20 PTO days."
    assert result["iterations"] == 2
    assert len(result["tools_used"]) == 1
    assert result["tools_used"][0]["tool_name"] == "SearchKnowledgeBase"
    assert result["tools_used"][0]["success"] is True
    assert len(result["sources"]) == 1


async def test_get_leave_balance_tool_is_excluded_from_caching(monkeypatch):
    async def _fake_get_leave_balance_tool(employee_id):
        return "You have 10 days remaining.", []

    monkeypatch.setattr(orchestration_agent, "get_leave_balance_tool", _fake_get_leave_balance_tool)

    tool_call_message = _ai_message(
        "", tool_calls=[{"name": "GetLeaveBalance", "args": {"input": ""}, "id": "call_1"}]
    )
    final_message = _ai_message("You have 10 days of PTO remaining.")
    bound_llm = FakeBoundLLM([tool_call_message, final_message])
    cache = FakeAnswerCache()
    _patch_common(monkeypatch, bound_llm, answer_cache=cache)

    await orchestration_agent.run_agent("what is my PTO balance", "EMP1")

    assert cache.store == {}


async def test_unknown_tool_name_reports_an_error_and_continues(monkeypatch):
    tool_call_message = _ai_message(
        "", tool_calls=[{"name": "NotARealTool", "args": {"input": "x"}, "id": "call_1"}]
    )
    final_message = _ai_message("I couldn't find that information.")
    bound_llm = FakeBoundLLM([tool_call_message, final_message])
    _patch_common(monkeypatch, bound_llm)

    result = await orchestration_agent.run_agent("do something unsupported", "EMP1")

    assert result["tools_used"][0]["success"] is False
    assert result["tools_used"][0]["tool_name"] == "NotARealTool"


async def test_iteration_limit_exhausted_returns_the_fallback_message(monkeypatch):
    tool_call_message = _ai_message(
        "", tool_calls=[{"name": "GetLeaveBalance", "args": {"input": ""}, "id": "call_1"}]
    )
    bound_llm = FakeBoundLLM([tool_call_message] * 3)

    async def _fake_get_leave_balance_tool(employee_id):
        return "balance text", []

    monkeypatch.setattr(orchestration_agent, "get_leave_balance_tool", _fake_get_leave_balance_tool)
    _patch_common(monkeypatch, bound_llm)

    result = await orchestration_agent.run_agent("what's my balance", "EMP1", max_iterations=3)

    assert "wasn't able to finish reasoning" in result["answer"]
    assert result["iterations"] == 3


async def test_cache_hit_returns_cached_answer_without_calling_the_llm(monkeypatch):
    bound_llm = FakeBoundLLM([])  # would raise IndexError (pop from empty) if ever called
    cache = FakeAnswerCache()
    _patch_common(monkeypatch, bound_llm, answer_cache=cache)

    cache_key = orchestration_agent._build_cache_key(
        query="what is the company name", pipeline="single-agentic-rag", max_iterations=5,
        search_strategy=None, role=None, department=None,
    )
    cache.store[cache_key] = {
        "answer": "Cached answer.", "tools_used": [], "tool_outputs": [], "sources": [],
        "iterations": 1, "llm_call_count": 1, "token_usage": None,
        "llm_context": {"turns": [], "mcp_tool_calls": []},
        "latency_ms": {"total": 1.0, "retrieval": None, "generation": None},
        "eval_scores": None, "guardrails": {"input": {"status": "pass"}, "output": {"status": "pass"}},
        "guardrails_model_used": None, "conversation_id": None,
    }

    result = await orchestration_agent.run_agent("what is the company name", "EMP1")

    assert result["answer"] == "Cached answer."
    assert result["served_from_cache"] is True
    assert result["llm_call_count"] == 0
