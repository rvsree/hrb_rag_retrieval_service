import time

from rag_retrieval.search_pre_processing import cache_layer


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


async def test_check_cache_returns_none_on_a_miss(monkeypatch):
    fake = FakeAnswerCache()
    monkeypatch.setattr(cache_layer, "_answer_cache", fake)
    result = await cache_layer.check_cache("key", "q", None, "EMP1", None, time.perf_counter())
    assert result is None


async def test_check_cache_returns_shaped_hit_on_a_hit(monkeypatch):
    fake = FakeAnswerCache()
    fake.store["key"] = {"answer": "cached answer", "tools_used": [], "eval_scores": None}
    monkeypatch.setattr(cache_layer, "_answer_cache", fake)

    result = await cache_layer.check_cache("key", "q", None, "EMP1", None, time.perf_counter())

    assert result["answer"] == "cached answer"
    assert result["served_from_cache"] is True
    assert result["llm_call_count"] == 0


async def test_write_cache_skips_entries_that_used_live_mcp_data(monkeypatch):
    fake = FakeAnswerCache()
    monkeypatch.setattr(cache_layer, "_answer_cache", fake)

    await cache_layer.write_cache(
        "key", "q", {"tools_used": [{"tool_name": "get_leave_balance"}], "conversation_id": None}
    )

    assert "key" not in fake.store


async def test_write_cache_stores_eligible_entries(monkeypatch):
    fake = FakeAnswerCache()
    monkeypatch.setattr(cache_layer, "_answer_cache", fake)

    await cache_layer.write_cache("key", "q", {"tools_used": [], "conversation_id": None})

    assert "key" in fake.store
