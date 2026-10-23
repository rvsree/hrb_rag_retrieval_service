from fastapi.testclient import TestClient

from rag_retrieval import hitl
from rag_retrieval.main import app

client = TestClient(app)


def test_submit_unpaid_leave_happy_path(monkeypatch):
    async def fake_run(**kwargs):
        return {
            "requestId": 1,
            "employeeId": kwargs["employee_id"],
            "leaveType": kwargs["leave_type_code"],
            "status": "PENDING_HITL",
            "isRepeatWithin12Months": False,
            "hitlRequestId": 7,
            "hitlStatus": "PENDING",
            "days": 3.0,
        }

    monkeypatch.setattr(hitl.routes, "run_unpaid_leave_request", fake_run)

    response = client.post(
        "/unpaid-leave/request",
        json={
            "user_profile": {"employee_id": "EMP052", "full_name": "Eddy Employee", "role": "employee"},
            "start_date": "2026-12-01",
            "end_date": "2026-12-03",
            "purpose": "Family caregiving",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hitl_request_context_id"] == 7
    assert body["days"] == 3.0


def test_submit_unpaid_leave_failure_returns_500(monkeypatch):
    async def fake_run(**kwargs):
        raise RuntimeError("mcp down")

    monkeypatch.setattr(hitl.routes, "run_unpaid_leave_request", fake_run)

    response = client.post(
        "/unpaid-leave/request",
        json={
            "user_profile": {"employee_id": "EMP052", "full_name": "Eddy Employee", "role": "employee"},
            "start_date": "2026-12-01",
            "end_date": "2026-12-03",
            "purpose": "Family caregiving",
        },
    )

    assert response.status_code == 500


def test_resolve_unpaid_leave_request(monkeypatch):
    captured = {}

    async def fake_resume(**kwargs):
        captured.update(kwargs)
        return {
            "hitlRequestId": kwargs["hitl_request_context_id"],
            "status": "RESOLVED",
            "decision": "APPROVED",
            "response": "ok",
            "relatedLeaveRequestUpdated": True,
        }

    monkeypatch.setattr(hitl.routes, "resume_unpaid_leave_decision", fake_resume)

    response = client.post(
        "/unpaid-leave/resolve",
        json={
            "user_profile": {"employee_id": "EMP021", "full_name": "Daniel Rodriguez", "role": "hr_support"},
            "hitl_request_context_id": 7,
            "decision": "APPROVED",
            "response_text": "ok",
        },
    )

    assert response.status_code == 200
    assert captured["resolved_by_employee_id"] == "EMP021"


def test_list_pending_unpaid_leave_requests(monkeypatch):
    async def fake_list():
        return {
            "total": 1,
            "requests": [
                {
                    "hitlRequestId": 7,
                    "requestType": "UNPAID_LEAVE_REQUEST",
                    "employeeId": "EMP052",
                    "query": "unpaid leave",
                    "context": {},
                    "status": "PENDING",
                    "assignedTo": None,
                    "createdAt": "2026-10-11T00:00:00",
                }
            ],
        }

    monkeypatch.setattr(hitl.routes, "list_pending_hitl_requests", fake_list)

    response = client.get("/unpaid-leave/pending")

    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_get_leave_document_download_url_returns_404_when_no_documents(monkeypatch):
    async def fake_list_documents(leave_request_id):
        return {"documents": []}

    monkeypatch.setattr(hitl.routes, "list_leave_documents", fake_list_documents)

    response = client.get("/unpaid-leave/documents/7")

    assert response.status_code == 404
