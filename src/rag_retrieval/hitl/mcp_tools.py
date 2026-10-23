"""HITL-specific MCP tool wrappers against hrb_lms_mcp - ported from
hrb_chatbot_service's ai/rag_pipeline/tools/mcp_tools/__init__.py, the
"Request Unpaid Leave" (Phase 162/165/168/172) subset only. Reuses
genai/mcp_tools.py's own _call()/_get_oauth_client() plumbing instead of
duplicating the OAuth2/MCP-session setup."""

import json

from rag_retrieval.genai.mcp_tools import _call

__all__ = [
    "get_leave_history_parsed",
    "submit_unpaid_leave_request",
    "resolve_hitl_request",
    "set_review_brief",
    "record_produced_artifact",
    "register_leave_document",
    "list_leave_documents",
    "list_my_leave_requests",
    "list_all_leave_requests",
    "list_pending_hitl_requests",
    "list_all_hitl_requests",
    "admin_delete_hitl_request",
]


async def _call_json(tool_name: str, arguments: dict) -> dict:
    """FastMCP returns one pretty-printed JSON text block per call - every
    wrapper below needs the parsed dict, not the raw text content list
    genai/mcp_tools.py's own _call() returns."""
    result = await _call(tool_name, arguments)
    return json.loads(result["content"][0])


async def get_leave_history_parsed(employee_id: str, limit: int = 50) -> dict:
    return await _call_json("get_leave_history", {"employee_id": employee_id, "limit": limit})


async def submit_unpaid_leave_request(
    employee_id: str,
    start_date: str,
    end_date: str,
    purpose: str,
    is_repeat_within_12_months: bool,
    leave_type_code: str = "UNPAID",
    document_reference: str | None = None,
) -> dict:
    return await _call_json(
        "submit_unpaid_leave_request",
        {
            "employee_id": employee_id,
            "start_date": start_date,
            "end_date": end_date,
            "purpose": purpose,
            "is_repeat_within_12_months": is_repeat_within_12_months,
            "leave_type_code": leave_type_code,
            "document_reference": document_reference,
        },
    )


async def resolve_hitl_request(
    hitl_request_context_id: int,
    decision: str,
    response_text: str,
    resolved_by: str | None = None,
    resolved_by_employee_id: str | None = None,
) -> dict:
    return await _call_json(
        "resolve_hitl_request",
        {
            "hitl_request_id": hitl_request_context_id,
            "decision": decision,
            "response_text": response_text,
            "resolved_by": resolved_by,
            "resolved_by_employee_id": resolved_by_employee_id,
        },
    )


async def set_review_brief(hitl_request_context_id: int, review_brief: str) -> dict:
    return await _call_json(
        "set_review_brief", {"hitl_request_id": hitl_request_context_id, "review_brief": review_brief}
    )


async def record_produced_artifact(hitl_request_context_id: int, artifact: dict) -> dict:
    return await _call_json(
        "record_produced_artifact", {"hitl_request_id": hitl_request_context_id, "artifact": artifact}
    )


async def register_leave_document(
    employee_id: str, s3_key: str, original_filename: str, content_type: str | None = None
) -> dict:
    return await _call_json(
        "register_leave_document",
        {
            "employee_id": employee_id,
            "s3_key": s3_key,
            "original_filename": original_filename,
            "content_type": content_type,
        },
    )


async def list_leave_documents(leave_request_id: int) -> dict:
    return await _call_json("list_leave_documents", {"leave_request_id": leave_request_id})


async def list_my_leave_requests(employee_id: str) -> dict:
    return await _call_json("list_my_leave_requests", {"employee_id": employee_id})


async def list_all_leave_requests(status: str | None = None) -> dict:
    return await _call_json("list_all_leave_requests", {"status": status})


async def list_pending_hitl_requests() -> dict:
    return await _call_json("list_hitl_requests", {"status": "PENDING", "request_type": "UNPAID_LEAVE_REQUEST"})


async def list_all_hitl_requests() -> dict:
    """The admin dashboard's own GET /admin/leave-requests - a different
    MCP tool (list_hitl_requests, status='all') from unpaid_leave.py's own
    GET /all route (list_all_leave_requests, a richer HR-specialist shape)
    - not the same call, don't conflate the two."""
    return await _call_json("list_hitl_requests", {"status": "all", "request_type": "UNPAID_LEAVE_REQUEST"})


async def admin_delete_hitl_request(hitl_request_context_id: int) -> dict:
    return await _call_json("admin_delete_hitl_request", {"hitl_request_id": hitl_request_context_id})
