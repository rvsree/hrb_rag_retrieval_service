"""The 3 retrieval-domain admin view route groups ported from
hrb_chatbot_service's api/admin/manage_admin.py - conversations, feedback,
pending HITL requests. Everything else in the original manage_admin.py
(interview-QA, build-deployment/CI, golden-dataset, MCP tool catalog,
admin-docs-sync-to-S3, request-metrics, FAQ submissions) is either already
ported to hrb_platform_components or a separate, still-orphan follow-up -
not this repo's concern.

Simplified, flagged: no RBAC, no rate limiting - same precedent every
other route in this repo already set."""

from fastapi import APIRouter, HTTPException

from rag_retrieval.core.stores import ConversationStore, FeedbackStore
from rag_retrieval.hitl.mcp_tools import admin_delete_hitl_request, list_all_hitl_requests
from rag_retrieval.hitl.s3_leave_docs import delete_leave_document_object
from rag_retrieval.models import (
    AdminConversationListResponse,
    AdminConversationSummary,
    AdminDeleteConversationResponse,
    AdminDeleteFeedbackResponse,
    AdminDeleteLeaveRequestResponse,
    ConversationDetailResponse,
    ConversationTurn,
    FeedbackListResponse,
    FeedbackRecord,
    HitlRequestSummary,
    PendingHitlRequestsResponse,
)

router = APIRouter(prefix="/admin", tags=["admin"])

_conversation_store = ConversationStore()
_feedback_store = FeedbackStore()


@router.get("/conversations", response_model=AdminConversationListResponse)
async def list_all_conversations():
    rows = _conversation_store.list_all_conversations()
    return AdminConversationListResponse(count=len(rows), conversations=[AdminConversationSummary(**row) for row in rows])


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_any_conversation(conversation_id: str):
    turns = _conversation_store.get_conversation_turns_any(conversation_id)
    return ConversationDetailResponse(conversation_id=conversation_id, turns=[ConversationTurn(**turn) for turn in turns])


@router.delete("/conversations/{conversation_id}", response_model=AdminDeleteConversationResponse)
async def delete_any_conversation(conversation_id: str):
    turns_deleted = _conversation_store.delete_conversation_any(conversation_id)
    return AdminDeleteConversationResponse(conversation_id=conversation_id, turns_deleted=turns_deleted)


@router.get("/feedback", response_model=FeedbackListResponse)
async def list_all_feedback():
    rows = _feedback_store.list_feedback(None)
    return FeedbackListResponse(count=len(rows), feedback=[FeedbackRecord(**row) for row in rows])


@router.delete("/feedback/{feedback_id}", response_model=AdminDeleteFeedbackResponse)
async def delete_feedback_entry(feedback_id: int):
    rows_deleted = _feedback_store.delete_feedback(feedback_id)
    return AdminDeleteFeedbackResponse(feedback_id=feedback_id, deleted=rows_deleted > 0)


@router.get("/leave-requests", response_model=PendingHitlRequestsResponse)
async def list_all_leave_requests_admin():
    try:
        data = await list_all_hitl_requests()
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not load leave requests. Please try again.") from error

    return PendingHitlRequestsResponse(
        total=data["total"],
        requests=[
            HitlRequestSummary(
                hitl_request_context_id=item["hitlRequestId"],
                request_type=item["requestType"],
                employee_id=item["employeeId"],
                query=item["query"],
                context=item["context"] or {},
                status=item["status"],
                assigned_to=item["assignedTo"],
                created_at=item["createdAt"],
            )
            for item in data["requests"]
        ],
    )


@router.delete("/leave-requests/{hitl_request_context_id}", response_model=AdminDeleteLeaveRequestResponse)
async def delete_any_leave_request(hitl_request_context_id: int):
    try:
        result = await admin_delete_hitl_request(hitl_request_context_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail="The delete could not be completed. Please try again.") from error

    for s3_key in result.get("deletedDocumentS3Keys", []):
        try:
            delete_leave_document_object(s3_key)
        except Exception:
            pass  # best-effort, same reasoning as the original

    return AdminDeleteLeaveRequestResponse(
        hitl_request_context_id=result["hitlRequestId"],
        deleted_hitl_request=result["deletedHitlRequest"],
        deleted_specialist_leave_reviews=result["deletedSpecialistLeaveReviews"],
        deleted_leave_request=result["deletedLeaveRequest"],
        deleted_document_count=len(result.get("deletedDocumentS3Keys", [])),
    )
