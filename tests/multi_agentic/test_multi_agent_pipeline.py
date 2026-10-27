"""Tests for the real LangGraph graph (Planner -> dispatch -> domain agents ->
Reviewer). The graph itself (StateGraph/Send) is real, trusted LangGraph
machinery - only this project's own agent-calling code is faked, matching
every other client test in this project."""

from rag_retrieval.core import guarded_pipeline
from rag_retrieval.multi_agentic import multi_agent_pipeline
from rag_retrieval.search_pre_processing import conversation_memory


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


def _patch_common(monkeypatch, answer_cache: FakeAnswerCache | None = None):
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

    monkeypatch.setattr(guarded_pipeline, "check_input", _passthrough_check_input)
    monkeypatch.setattr(guarded_pipeline, "check_output", _passthrough_check_output)
    monkeypatch.setattr(conversation_memory, "load_history", _fake_load_history)
    monkeypatch.setattr(conversation_memory, "save_turn", _fake_save_turn)
    monkeypatch.setattr(multi_agent_pipeline, "normalize_query", _passthrough_normalize)
    monkeypatch.setattr(multi_agent_pipeline, "check_input", _passthrough_check_input)
    monkeypatch.setattr(multi_agent_pipeline, "_answer_cache", answer_cache or FakeAnswerCache())
    # _score_live_answer's own wider-pool fetch (only triggered when sources
    # is non-empty) calls search_knn directly - same fake test_orchestration_agent.py
    # already uses for the same reason.
    monkeypatch.setattr(multi_agent_pipeline, "search_knn", lambda query, top_k, vector_db: [])


async def test_single_task_dispatch_answers_via_reviewer(monkeypatch):
    _patch_common(monkeypatch)

    async def _fake_plan(query, request_id=None, planner_model=None):
        return [{"agent": "vector_kb_agent", "focus": query, "reasoning": "a policy question"}], {
            "label": "planner", "latency_ms": 1.0, "system_prompt": "p", "human_message": query,
            "chat_history": [], "response": "...",
        }

    async def _fake_vector_kb_run(focus, search_strategy=None, role=None, department=None):
        return "Found: 401k matches up to 5%.", [
            {"document_id": "doc1", "chunk_index": 0, "filename": "401k.pdf", "text": "matches up to 5%", "score": 0.9}
        ]

    async def _fake_review(query, agent_results, request_id=None):
        assert len(agent_results) == 1
        return "AmazingBank matches your 401k contribution up to 5%.", {
            "label": "reviewer", "latency_ms": 1.0, "system_prompt": "p", "human_message": "h",
            "chat_history": [], "response": "...",
        }

    async def _fake_follow_ups(query, answer):
        return ["How do I enroll?"]

    monkeypatch.setattr(multi_agent_pipeline.planner_agent, "plan", _fake_plan)
    monkeypatch.setattr(multi_agent_pipeline.vector_kb_agent, "run", _fake_vector_kb_run)
    monkeypatch.setattr(multi_agent_pipeline, "review", _fake_review)
    monkeypatch.setattr(multi_agent_pipeline, "generate_follow_ups", _fake_follow_ups)

    result = await multi_agent_pipeline.run_multi_agent("what is the 401k match", "EMP1")

    assert result["answer"] == "AmazingBank matches your 401k contribution up to 5%."
    assert result["tasks"] == [{"agent": "vector_kb_agent", "focus": "what is the 401k match", "reasoning": "a policy question"}]
    assert result["follow_up_questions"] == ["How do I enroll?"]
    assert result["iterations"] == 1
    assert len(result["sources"]) == 1
    assert result["served_from_cache"] is False


async def test_two_task_dispatch_merges_both_agent_results(monkeypatch):
    _patch_common(monkeypatch)

    async def _fake_plan(query, request_id=None, planner_model=None):
        return [
            {"agent": "vector_kb_agent", "focus": "401k policy", "reasoning": "policy part"},
            {"agent": "lms_ops_agent", "focus": "my leave balance", "reasoning": "live data part"},
        ], {"label": "planner", "latency_ms": 1.0, "system_prompt": "p", "human_message": query, "chat_history": [], "response": "..."}

    async def _fake_vector_kb_run(focus, search_strategy=None, role=None, department=None):
        return "policy answer", []

    async def _fake_lms_ops_run(focus, employee_id):
        return "You have 10 days remaining.", []

    async def _fake_review(query, agent_results, request_id=None):
        assert len(agent_results) == 2
        return "merged answer", None

    async def _fake_follow_ups(query, answer):
        return []

    monkeypatch.setattr(multi_agent_pipeline.planner_agent, "plan", _fake_plan)
    monkeypatch.setattr(multi_agent_pipeline.vector_kb_agent, "run", _fake_vector_kb_run)
    monkeypatch.setattr(multi_agent_pipeline.lms_ops_agent, "run", _fake_lms_ops_run)
    monkeypatch.setattr(multi_agent_pipeline, "review", _fake_review)
    monkeypatch.setattr(multi_agent_pipeline, "generate_follow_ups", _fake_follow_ups)

    result = await multi_agent_pipeline.run_multi_agent("401k policy and my leave balance", "EMP1")

    assert result["answer"] == "merged answer"
    assert result["iterations"] == 2
    assert {task["agent"] for task in result["tasks"]} == {"vector_kb_agent", "lms_ops_agent"}


async def test_dispatch_including_a_live_data_agent_is_excluded_from_caching(monkeypatch):
    cache = FakeAnswerCache()
    _patch_common(monkeypatch, answer_cache=cache)

    async def _fake_plan(query, request_id=None, planner_model=None):
        return [{"agent": "lms_ops_agent", "focus": query, "reasoning": "live data"}], None

    async def _fake_lms_ops_run(focus, employee_id):
        return "You have 10 days remaining.", []

    async def _fake_review(query, agent_results, request_id=None):
        return "You have 10 days remaining.", None

    async def _fake_follow_ups(query, answer):
        return []

    monkeypatch.setattr(multi_agent_pipeline.planner_agent, "plan", _fake_plan)
    monkeypatch.setattr(multi_agent_pipeline.lms_ops_agent, "run", _fake_lms_ops_run)
    monkeypatch.setattr(multi_agent_pipeline, "review", _fake_review)
    monkeypatch.setattr(multi_agent_pipeline, "generate_follow_ups", _fake_follow_ups)

    await multi_agent_pipeline.run_multi_agent("what is my leave balance", "EMP1")

    assert cache.store == {}


async def test_dispatch_with_only_vector_kb_agent_is_cache_eligible(monkeypatch):
    cache = FakeAnswerCache()
    _patch_common(monkeypatch, answer_cache=cache)

    async def _fake_plan(query, request_id=None, planner_model=None):
        return [{"agent": "vector_kb_agent", "focus": query, "reasoning": "policy"}], None

    async def _fake_vector_kb_run(focus, search_strategy=None, role=None, department=None):
        return "policy answer", []

    async def _fake_review(query, agent_results, request_id=None):
        return "policy answer", None

    async def _fake_follow_ups(query, answer):
        return []

    monkeypatch.setattr(multi_agent_pipeline.planner_agent, "plan", _fake_plan)
    monkeypatch.setattr(multi_agent_pipeline.vector_kb_agent, "run", _fake_vector_kb_run)
    monkeypatch.setattr(multi_agent_pipeline, "review", _fake_review)
    monkeypatch.setattr(multi_agent_pipeline, "generate_follow_ups", _fake_follow_ups)

    await multi_agent_pipeline.run_multi_agent("what is the 401k policy", "EMP1")

    assert len(cache.store) == 1


async def test_cache_hit_returns_cached_answer_without_calling_the_planner(monkeypatch):
    cache = FakeAnswerCache()
    _patch_common(monkeypatch, answer_cache=cache)

    async def _fake_plan(query, request_id=None, planner_model=None):
        raise AssertionError("planner should not be called on a cache hit")

    monkeypatch.setattr(multi_agent_pipeline.planner_agent, "plan", _fake_plan)

    cache_key = multi_agent_pipeline._build_cache_key(
        query="what is the 401k policy", pipeline="multi-agentic-rag", search_strategy=None, role=None, department=None,
    )
    cache.store[cache_key] = {
        "answer": "Cached answer.", "tasks": [], "follow_up_questions": [], "tools_used": [],
        "sources": [], "iterations": 0, "guardrails": {"input": {"status": "pass"}, "output": {"status": "pass"}},
        "guardrails_model_used": None, "eval_scores": None,
        "latency_ms": {"total": 1.0, "retrieval": None, "generation": None, "eval": None},
        "conversation_id": None,
    }

    result = await multi_agent_pipeline.run_multi_agent("what is the 401k policy", "EMP1")

    assert result["answer"] == "Cached answer."
    assert result["served_from_cache"] is True
    assert result["llm_call_count"] == 0
