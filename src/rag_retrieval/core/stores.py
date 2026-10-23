"""Postgres/Redis-backed stores this service's own pipeline needs - ported
from hrb_chatbot_service's db_gateway-combined store classes, minimized to
the methods genai-rag's own call path actually uses (not every admin-view
method the originals have - those stay out of scope here, same "flag, don't
silently port everything" judgment rag_ingestion_service's own Phase 1 used).

DocumentMetadataStore reads from hrb_rag_ingestion.documents - the SAME
schema rag_ingestion_service already owns (Phase 251's local migration).
A real cross-service data coupling, not yet resolved behind a real API
boundary between the two services - flagged in the delivery note, not
silently decided as a permanent design."""

import json

from hrb_common.observability.logger import get_logger

from rag_retrieval.gateway import get_postgres_connection, get_redis

logger = get_logger("core.stores")


class DocumentMetadataStore:
    """Read-only from this service's point of view - ingestion owns writes."""

    def get_document(self, document_id: str) -> dict | None:
        try:
            conn = get_postgres_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, filename, doc_description FROM hrb_rag_ingestion.documents WHERE id = %s",
                    (document_id,),
                )
                row = cur.fetchone()
        except Exception as error:
            logger.warning("get_document(%r) failed, treating as unknown: %s", document_id, error)
            return None
        if row is None:
            return None
        return {"id": row[0], "filename": row[1], "doc_description": row[2]}

    def list_documents(self) -> list[dict]:
        try:
            conn = get_postgres_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, filename, doc_description, department, restricted_roles, is_current "
                    "FROM hrb_rag_ingestion.documents"
                )
                rows = cur.fetchall()
        except Exception as error:
            logger.warning("list_documents() failed, returning empty: %s", error)
            return []
        return [
            {
                "id": row[0],
                "filename": row[1],
                "doc_description": row[2],
                "department": row[3],
                "restricted_roles": row[4],
                "is_current": row[5],
            }
            for row in rows
        ]


class ConversationStore:
    """Postgres system-of-record for conversation turns - hrb_chat_retrieval
    schema (this service's own, per Phase 251's local mapping)."""

    TABLE = "hrb_rag_retrieval.conversation_turns"

    def _ensure_table(self, cur) -> None:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                employee_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

    def save_turn(self, conversation_id: str, employee_id: str | None, role: str, content: str) -> None:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            # nosec B608 - self.TABLE is a hardcoded class constant, never
            # user input; the real values go through real %s placeholders.
            cur.execute(
                f"INSERT INTO {self.TABLE} (conversation_id, employee_id, role, content) VALUES (%s, %s, %s, %s)",  # nosec B608
                (conversation_id, employee_id, role, content),
            )
        conn.commit()

    def load_turns(self, conversation_id: str) -> list[dict]:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                f"SELECT role, content FROM {self.TABLE} WHERE conversation_id = %s ORDER BY id",  # nosec B608
                (conversation_id,),
            )
            rows = cur.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def delete_conversation(self, conversation_id: str, employee_id: str) -> int:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                f"DELETE FROM {self.TABLE} WHERE conversation_id = %s AND employee_id = %s",  # nosec B608
                (conversation_id, employee_id),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted

    def list_all_conversations(self) -> list[dict]:
        """Phase 4 - admin-only: every conversation across every employee,
        newest-activity-first. Deliberately not scoped to one employee_id -
        the one legitimate "see everyone's" case in this store."""
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                f"SELECT conversation_id, employee_id, role, content, created_at "  # nosec B608
                f"FROM {self.TABLE} ORDER BY id ASC"
            )
            rows = cur.fetchall()

        conversations: dict[str, dict] = {}
        for conversation_id, employee_id, _role, content, created_at in rows:
            if conversation_id not in conversations:
                conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "employee_id": employee_id,
                    "title": content,
                    "started_at": created_at,
                    "last_updated_at": created_at,
                }
            else:
                conversations[conversation_id]["last_updated_at"] = created_at

        ordered = sorted(conversations.values(), key=lambda c: c["last_updated_at"], reverse=True)
        for conversation in ordered:
            conversation["started_at"] = conversation["started_at"].isoformat()
            conversation["last_updated_at"] = conversation["last_updated_at"].isoformat()
        return ordered

    def get_conversation_turns_any(self, conversation_id: str) -> list[dict]:
        """Phase 4 - admin-only: the full transcript for any conversation
        regardless of owner."""
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                f"SELECT role, content, created_at FROM {self.TABLE} "  # nosec B608
                f"WHERE conversation_id = %s ORDER BY id ASC",
                (conversation_id,),
            )
            rows = cur.fetchall()
        return [{"role": role, "content": content, "created_at": created_at.isoformat()} for role, content, created_at in rows]

    def delete_conversation_any(self, conversation_id: str) -> int:
        """Phase 4 - admin-only: deletes every turn for one conversation_id
        regardless of which employee it belongs to."""
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(
                f"DELETE FROM {self.TABLE} WHERE conversation_id = %s",  # nosec B608
                (conversation_id,),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted


class StmCache:
    """Redis cache-aside in front of ConversationStore - a performance cache
    only, degrades to a Postgres read on a miss, never the other way."""

    def __init__(self, ttl_seconds: int = 7200):
        self.ttl_seconds = ttl_seconds

    def _key(self, conversation_id: str) -> str:
        return f"rag_retrieval:stm:{conversation_id}"

    def get_turns(self, conversation_id: str) -> list[dict] | None:
        try:
            raw = get_redis().get(self._key(conversation_id))
        except Exception as error:
            logger.warning("StmCache.get_turns failed, treating as a miss: %s", error)
            return None
        return json.loads(raw) if raw else None

    def backfill(self, conversation_id: str, turns: list[dict]) -> None:
        try:
            get_redis().setex(self._key(conversation_id), self.ttl_seconds, json.dumps(turns))
        except Exception as error:
            logger.warning("StmCache.backfill failed, skipping: %s", error)

    def append_turn(self, conversation_id: str, role: str, content: str) -> None:
        try:
            existing = self.get_turns(conversation_id) or []
            existing.append({"role": role, "content": content})
            get_redis().setex(self._key(conversation_id), self.ttl_seconds, json.dumps(existing))
        except Exception as error:
            logger.warning("StmCache.append_turn failed, skipping: %s", error)

    def clear(self, conversation_id: str) -> None:
        try:
            get_redis().delete(self._key(conversation_id))
        except Exception as error:
            logger.warning("StmCache.clear failed, skipping: %s", error)


class AnswerCacheStore:
    """Redis-backed exact-match answer cache - the real current
    implementation (hrb_chatbot_service's own Phase 77/84 moved this off
    Postgres; the old answer_cache/embedding_cache Postgres tables are
    orphaned, confirmed during rag_ingestion_service's own Phase 251 work)."""

    def __init__(self, ttl_seconds: int = 21600):
        self.ttl_seconds = ttl_seconds

    def _key(self, cache_key: str) -> str:
        return f"rag_retrieval:answer_cache:{cache_key}"

    def get(self, cache_key: str) -> dict | None:
        try:
            raw = get_redis().get(self._key(cache_key))
        except Exception as error:
            logger.warning("AnswerCacheStore.get failed, treating as a miss: %s", error)
            return None
        return json.loads(raw) if raw else None

    def set(self, cache_key: str, checked_query: str, result: dict) -> None:
        try:
            get_redis().setex(self._key(cache_key), self.ttl_seconds, json.dumps(result, default=str))
        except Exception as error:
            logger.warning("AnswerCacheStore.set failed, skipping: %s", error)

    def tag_conversation(self, conversation_id: str, cache_key: str) -> None:
        """Best-effort only this phase - purge-on-conversation-delete isn't
        wired up yet (flagged, not a silent gap: the original's own
        conversation-delete cascade into the answer cache isn't ported)."""
        return


class FeedbackStore:
    """Phase 4 - replaces the Phase 1 stub with a real implementation, now
    that the admin feedback views need list/delete. get_latest_feedback_for_conversation()
    is the one method genai-rag's own context_assembly.py needs (still not
    wired into genai-rag's own route this phase - only the separate,
    out-of-scope auto-router endpoint calls it)."""

    TABLE = "hrb_chat_feedback.feedback"

    def _ensure_table(self, cur) -> None:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id SERIAL PRIMARY KEY,
                employee_id TEXT NOT NULL,
                conversation_id TEXT,
                message_id TEXT NOT NULL,
                vote TEXT NOT NULL,
                reason_tags TEXT NOT NULL,
                notes TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    def get_latest_feedback_for_conversation(self, conversation_id: str | None) -> dict | None:
        if not conversation_id:
            return None
        try:
            conn = get_postgres_connection()
            with conn.cursor() as cur:
                self._ensure_table(cur)
                cur.execute(
                    f"SELECT vote, notes FROM {self.TABLE} "  # nosec B608
                    f"WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 1",
                    (conversation_id,),
                )
                row = cur.fetchone()
        except Exception as error:
            logger.warning("get_latest_feedback_for_conversation(%r) failed, treating as none: %s", conversation_id, error)
            return None
        return {"vote": row[0], "notes": row[1]} if row else None

    def list_feedback(self, employee_id: str | None) -> list[dict]:
        """employee_id=None lists every caller's feedback - admin-only."""
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            if employee_id is None:
                cur.execute(
                    f"SELECT id, employee_id, conversation_id, message_id, vote, reason_tags, notes, "  # nosec B608
                    f"question, answer, created_at FROM {self.TABLE} ORDER BY id DESC"
                )
            else:
                cur.execute(
                    f"SELECT id, employee_id, conversation_id, message_id, vote, reason_tags, notes, "  # nosec B608
                    f"question, answer, created_at FROM {self.TABLE} WHERE employee_id = %s ORDER BY id DESC",
                    (employee_id,),
                )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "employee_id": row[1],
                "conversation_id": row[2],
                "message_id": row[3],
                "vote": row[4],
                "reason_tags": json.loads(row[5]),
                "notes": row[6],
                "question": row[7],
                "answer": row[8],
                "created_at": row[9].isoformat(),
            }
            for row in rows
        ]

    def delete_feedback(self, feedback_id: int) -> int:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            self._ensure_table(cur)
            cur.execute(f"DELETE FROM {self.TABLE} WHERE id = %s", (feedback_id,))  # nosec B608
            deleted = cur.rowcount
        conn.commit()
        return deleted
