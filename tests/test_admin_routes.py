from fastapi.testclient import TestClient

from rag_retrieval import admin
from rag_retrieval.main import app

client = TestClient(app)


class _FakeConversationStore:
    def list_all_conversations(self):
        return [
            {
                "conversation_id": "c1",
                "employee_id": "EMP052",
                "title": "hello",
                "started_at": "2026-10-11T00:00:00",
                "last_updated_at": "2026-10-11T00:05:00",
            }
        ]

    def get_conversation_turns_any(self, conversation_id):
        return [{"role": "human", "content": "hi", "created_at": "2026-10-11T00:00:00"}]

    def delete_conversation_any(self, conversation_id):
        return 2


class _FakeFeedbackStore:
    def list_feedback(self, employee_id):
        return [
            {
                "id": 1,
                "employee_id": "EMP052",
                "conversation_id": "c1",
                "message_id": "m1",
                "vote": "up",
                "reason_tags": [],
                "notes": None,
                "question": "q",
                "answer": "a",
                "created_at": "2026-10-11T00:00:00",
            }
        ]

    def delete_feedback(self, feedback_id):
        return 1


def test_list_all_conversations(monkeypatch):
    monkeypatch.setattr(admin.routes, "_conversation_store", _FakeConversationStore())

    response = client.get("/admin/conversations")

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_get_any_conversation(monkeypatch):
    monkeypatch.setattr(admin.routes, "_conversation_store", _FakeConversationStore())

    response = client.get("/admin/conversations/c1")

    assert response.status_code == 200
    assert response.json()["turns"][0]["content"] == "hi"


def test_delete_any_conversation(monkeypatch):
    monkeypatch.setattr(admin.routes, "_conversation_store", _FakeConversationStore())

    response = client.delete("/admin/conversations/c1")

    assert response.status_code == 200
    assert response.json()["turns_deleted"] == 2


def test_list_all_feedback(monkeypatch):
    monkeypatch.setattr(admin.routes, "_feedback_store", _FakeFeedbackStore())

    response = client.get("/admin/feedback")

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_delete_feedback_entry(monkeypatch):
    monkeypatch.setattr(admin.routes, "_feedback_store", _FakeFeedbackStore())

    response = client.delete("/admin/feedback/1")

    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_list_all_leave_requests_admin(monkeypatch):
    async def fake_list_all():
        return {
            "total": 1,
            "requests": [
                {
                    "hitlRequestId": 7,
                    "requestType": "UNPAID_LEAVE_REQUEST",
                    "employeeId": "EMP052",
                    "query": "unpaid leave",
                    "context": {},
                    "status": "RESOLVED",
                    "assignedTo": "EMP021",
                    "createdAt": "2026-10-11T00:00:00",
                }
            ],
        }

    monkeypatch.setattr(admin.routes, "list_all_hitl_requests", fake_list_all)

    response = client.get("/admin/leave-requests")

    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_delete_any_leave_request(monkeypatch):
    async def fake_delete(hitl_request_context_id):
        return {
            "hitlRequestId": hitl_request_context_id,
            "deletedHitlRequest": True,
            "deletedSpecialistLeaveReviews": 1,
            "deletedLeaveRequest": True,
            "deletedDocumentS3Keys": ["key1"],
        }

    def fake_delete_object(s3_key):
        return None

    monkeypatch.setattr(admin.routes, "admin_delete_hitl_request", fake_delete)
    monkeypatch.setattr(admin.routes, "delete_leave_document_object", fake_delete_object)

    response = client.delete("/admin/leave-requests/7")

    assert response.status_code == 200
    body = response.json()
    assert body["deleted_document_count"] == 1
