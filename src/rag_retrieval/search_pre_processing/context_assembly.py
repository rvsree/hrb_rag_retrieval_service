"""Search Pre-Processing Layer's own entry point - ported from
hrb_chatbot_service's ai/search_pre_processing/context_assembly.py. Not
wired into genai-rag's own route this phase - only the separate
auto-router endpoint calls this, which is out of this phase's scope.
Ported for completeness per the spec's own "port in full" instruction."""

from typing import Literal

from langchain_core.messages import BaseMessage

from rag_retrieval.core.stores import FeedbackStore
from rag_retrieval.search_pre_processing import conversation_memory
from rag_retrieval.search_pre_processing.mode_router import classify_query_complexity

_feedback_store = FeedbackStore()


class RequestContext:
    def __init__(
        self,
        user_profile,
        conversation_id: str | None,
        query: str,
        stm: list[BaseMessage],
        feedback: dict | None,
        classification: Literal["single", "multi"],
    ):
        self.user_profile = user_profile
        self.conversation_id = conversation_id
        self.query = query
        self.stm = stm
        self.feedback = feedback
        self.classification = classification


async def assemble_context(user_profile, conversation_id: str | None, query: str, router_model: str | None = None) -> RequestContext:
    stm = await conversation_memory.load_history(conversation_id) if conversation_id else []
    feedback = _feedback_store.get_latest_feedback_for_conversation(conversation_id)
    classification = await classify_query_complexity(query, router_model)

    return RequestContext(user_profile, conversation_id, query, stm, feedback, classification)
