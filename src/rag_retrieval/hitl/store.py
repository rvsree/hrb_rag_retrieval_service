"""Maps a HITL request's hitl_request_context_id to the LangGraph thread_id
its originating "Request Unpaid Leave" graph run is checkpointed under, so
a later /resolve call (which only knows hitl_request_context_id, the
existing API contract) can find the right paused thread to resume. Ported
from hrb_chatbot_service's agent_thread_store.py, connection plumbing
swapped to this repo's own gateway.get_postgres_connection()."""

from hrb_common.observability.logger import get_logger

from rag_retrieval.gateway import get_postgres_connection

logger = get_logger("hitl.store")


class AgentThreadStore:
    TABLE = "hrb_chat_hitl.agent_threads"

    def _ensure_table(self, cur) -> None:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                hitl_request_context_id INTEGER PRIMARY KEY,
                agent_thread_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    def save_thread_mapping(self, hitl_request_context_id: int, agent_thread_id: str) -> None:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                # nosec B608 - self.TABLE is a hardcoded class constant, never user input.
                f"INSERT INTO {self.TABLE} (hitl_request_context_id, agent_thread_id) VALUES (%s, %s) "  # nosec B608
                "ON CONFLICT (hitl_request_context_id) DO UPDATE SET agent_thread_id = EXCLUDED.agent_thread_id",
                (hitl_request_context_id, agent_thread_id),
            )
        conn.commit()

    def get_thread_id(self, hitl_request_context_id: int) -> str | None:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                f"SELECT agent_thread_id FROM {self.TABLE} WHERE hitl_request_context_id = %s",  # nosec B608
                (hitl_request_context_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None
