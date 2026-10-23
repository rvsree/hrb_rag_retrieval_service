from rag_retrieval.search_pre_processing import conversation_memory


class FakeConversationStore:
    def __init__(self):
        self.turns: dict[str, list[dict]] = {}

    def save_turn(self, conversation_id, employee_id, role, content):
        self.turns.setdefault(conversation_id, []).append({"role": role, "content": content})

    def load_turns(self, conversation_id):
        return self.turns.get(conversation_id, [])

    def delete_conversation(self, conversation_id, employee_id):
        existed = conversation_id in self.turns
        self.turns.pop(conversation_id, None)
        return 1 if existed else 0


class FakeStmCache:
    def __init__(self):
        self.cache: dict[str, list[dict]] = {}

    def get_turns(self, conversation_id):
        return self.cache.get(conversation_id)

    def backfill(self, conversation_id, turns):
        self.cache[conversation_id] = turns

    def append_turn(self, conversation_id, role, content):
        self.cache.setdefault(conversation_id, []).append({"role": role, "content": content})

    def clear(self, conversation_id):
        self.cache.pop(conversation_id, None)


async def test_load_history_falls_back_to_postgres_on_a_redis_miss(monkeypatch):
    store = FakeConversationStore()
    stm = FakeStmCache()
    store.turns["conv-1"] = [{"role": "human", "content": "hi"}]
    monkeypatch.setattr(conversation_memory, "_conversation_store", store)
    monkeypatch.setattr(conversation_memory, "_stm_cache", stm)

    messages = await conversation_memory.load_history("conv-1")

    assert len(messages) == 1
    assert messages[0].content == "hi"
    assert "conv-1" in stm.cache  # backfilled


async def test_save_turn_writes_to_both_postgres_and_redis(monkeypatch):
    store = FakeConversationStore()
    stm = FakeStmCache()
    monkeypatch.setattr(conversation_memory, "_conversation_store", store)
    monkeypatch.setattr(conversation_memory, "_stm_cache", stm)

    await conversation_memory.save_turn("conv-1", "EMP1", "hello", "hi there")

    assert len(store.turns["conv-1"]) == 2
    assert len(stm.cache["conv-1"]) == 2


def test_new_conversation_id_is_unique():
    assert conversation_memory.new_conversation_id() != conversation_memory.new_conversation_id()
