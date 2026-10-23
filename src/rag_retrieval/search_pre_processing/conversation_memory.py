"""Server-side conversation history for multi-turn support - ported from
hrb_chatbot_service's ai/search_pre_processing/conversation_memory.py,
adapted to this repo's own core/stores.py (ConversationStore/StmCache)
instead of a combined db_gateway. Postgres stays the system of record;
Redis (StmCache) is a cache-aside layer in front of it, same as the
original."""

import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag_retrieval.core.stores import ConversationStore, StmCache

_conversation_store = ConversationStore()
_stm_cache = StmCache()


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def _turns_to_messages(turns: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in turns:
        if turn["role"] == "human":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


async def load_history(conversation_id: str) -> list[BaseMessage]:
    """Returns [] for an unknown/new conversation_id - never raises. Reads
    STM (Redis) first; on a miss, falls back to Postgres and backfills Redis."""
    cached_turns = _stm_cache.get_turns(conversation_id)
    if cached_turns is not None:
        return _turns_to_messages(cached_turns)

    turns = _conversation_store.load_turns(conversation_id)
    _stm_cache.backfill(conversation_id, turns)
    return _turns_to_messages(turns)


async def save_turn(conversation_id: str, employee_id: str | None, human_content: str, ai_content: str) -> None:
    _conversation_store.save_turn(conversation_id, employee_id, "human", human_content)
    _conversation_store.save_turn(conversation_id, employee_id, "ai", ai_content)
    _stm_cache.append_turn(conversation_id, "human", human_content)
    _stm_cache.append_turn(conversation_id, "ai", ai_content)


async def delete_conversation(conversation_id: str, employee_id: str) -> int:
    deleted = _conversation_store.delete_conversation(conversation_id, employee_id)
    if deleted:
        _stm_cache.clear(conversation_id)
    return deleted
