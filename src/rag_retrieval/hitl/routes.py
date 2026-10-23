"""Phase 162/165 - "Request Unpaid Leave". Ported from hrb_chatbot_service's
api/leave/unpaid_leave.py - 7 real routes.

Simplified, flagged: no RBAC, no rate limiting - same "accepts requests
with no identity check for now" precedent this repo's own routes.py
(Phase 1) already set. Ownership/role checks the original enforced via
require_role() are dropped; employee_id is still taken from the real
payload/query param, just not authorization-checked against the caller's
asserted role."""

import uuid

from fastapi import APIRouter, HTTPException, Query

from rag_retrieval.hitl.mcp_tools import (
    list_all_leave_requests,
    list_leave_documents,
    list_my_leave_requests,
    list_pending_hitl_requests,
    register_leave_document,
)
from rag_retrieval.hitl.s3_leave_docs import (
    generate_leave_document_download_url,
    generate_leave_document_upload_url,
)
from rag_retrieval.hitl.unpaid_leave_agent import resume_unpaid_leave_decision, run_unpaid_leave_request
from rag_retrieval.models import (
    HitlRequestSummary,
    HrLeaveRequestsResponse,
    HrLeaveRequestSummary,
    LeaveDocumentDownloadUrlResponse,
    LeaveDocumentUploadUrlRequest,
    LeaveDocumentUploadUrlResponse,
    MyLeaveRequestsResponse,
    MyLeaveRequestSummary,
    PendingHitlRequestsResponse,
    ResolveHitlRequestBody,
    ResolveHitlRequestResponse,
    UnpaidLeaveRequestBody,
    UnpaidLeaveRequestResponse,
)

router = APIRouter(prefix="/unpaid-leave", tags=["unpaid-leave"])


@router.post("/upload-url", response_model=LeaveDocumentUploadUrlResponse)
async def get_leave_document_upload_url(payload: LeaveDocumentUploadUrlRequest):
    employee_id = payload.user_profile.employee_id
    token = uuid.uuid4().hex
    result = generate_leave_document_upload_url(token, payload.filename, payload.content_type)

    try:
        await register_leave_document(employee_id, result["key"], payload.filename, payload.content_type)
    except Exception:
        pass  # best-effort, same reasoning as the original - the upload URL is already returned either way

    return LeaveDocumentUploadUrlResponse(**result)


@router.post("/request", response_model=UnpaidLeaveRequestResponse)
async def submit_unpaid_leave(payload: UnpaidLeaveRequestBody):
    try:
        result = await run_unpaid_leave_request(
            employee_id=payload.user_profile.employee_id,
            start_date=payload.start_date,
            end_date=payload.end_date,
            purpose=payload.purpose,
            leave_type_code=payload.leave_type_code.value,
            document_reference=payload.document_reference,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="The leave request could not be submitted. Please try again.") from error

    return UnpaidLeaveRequestResponse(
        request_id=result["requestId"],
        employee_id=result["employeeId"],
        leave_type=result["leaveType"],
        status=result["status"],
        is_repeat_within_12_months=result["isRepeatWithin12Months"],
        hitl_request_context_id=result["hitlRequestId"],
        hitl_status=result["hitlStatus"],
        days=result["days"],
    )


@router.get("/pending", response_model=PendingHitlRequestsResponse)
async def list_pending_unpaid_leave_requests():
    try:
        data = await list_pending_hitl_requests()
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not load pending leave requests. Please try again.") from error

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


@router.get("/all", response_model=HrLeaveRequestsResponse)
async def list_all_unpaid_leave_requests():
    try:
        data = await list_all_leave_requests(status="all")
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not load leave requests. Please try again.") from error

    return HrLeaveRequestsResponse(
        total=data["total"],
        requests=[
            HrLeaveRequestSummary(
                hitl_request_context_id=item["hitlRequestId"],
                request_id=item["requestId"],
                employee_id=item["employeeId"],
                employee_name=item["employeeName"],
                leave_type=item["leaveType"],
                start_date=item["startDate"],
                end_date=item["endDate"],
                days=item["days"],
                purpose=item["purpose"],
                status=item["status"],
                decision=item["decision"],
                hr_comments=item["hrComments"],
                hr_specialist_name=item["hrSpecialistName"],
                created_at=item["createdAt"],
                responded_at=item["respondedAt"],
                task_state=item.get("taskState"),
                review_brief=item.get("reviewBrief"),
                decision_message=item.get("decisionMessage"),
                produced_artifacts=item.get("producedArtifacts") or [],
            )
            for item in data["requests"]
        ],
    )


@router.post("/resolve", response_model=ResolveHitlRequestResponse)
async def resolve_unpaid_leave_request(payload: ResolveHitlRequestBody):
    try:
        result = await resume_unpaid_leave_decision(
            hitl_request_context_id=payload.hitl_request_context_id,
            decision=payload.decision,
            response_text=payload.response_text,
            resolved_by=payload.user_profile.full_name,
            resolved_by_employee_id=payload.user_profile.employee_id,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="The decision could not be recorded. Please try again.") from error

    return ResolveHitlRequestResponse(
        hitl_request_context_id=result["hitlRequestId"],
        status=result["status"],
        decision=result["decision"],
        response=result["response"],
        related_leave_request_updated=result["relatedLeaveRequestUpdated"],
    )


@router.get("/my-requests", response_model=MyLeaveRequestsResponse)
async def list_my_unpaid_leave_requests(employee_id: str = Query(...)):
    try:
        data = await list_my_leave_requests(employee_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not load your leave requests. Please try again.") from error

    return MyLeaveRequestsResponse(
        total=data["total"],
        requests=[
            MyLeaveRequestSummary(
                request_id=item["requestId"],
                leave_type=item["leaveType"],
                start_date=item["startDate"],
                end_date=item["endDate"],
                days=item["days"],
                purpose=item["purpose"],
                status=item["status"],
                decision=item["decision"],
                hr_comments=item["hrComments"],
                hr_specialist_name=item["hrSpecialistName"],
                created_at=item["createdAt"],
                responded_at=item["respondedAt"],
                task_state=item.get("taskState"),
                review_brief=item.get("reviewBrief"),
                decision_message=item.get("decisionMessage"),
                produced_artifacts=item.get("producedArtifacts") or [],
            )
            for item in data["requests"]
        ],
    )


@router.get("/documents/{leave_request_id}", response_model=LeaveDocumentDownloadUrlResponse)
async def get_leave_document_download_url(leave_request_id: int):
    """Ownership/role check dropped (no RBAC in this repo) - every caller
    can reach any leave_request_id's document, same flagged simplification
    as every other route here."""
    try:
        data = await list_leave_documents(leave_request_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not load the document. Please try again.") from error

    documents = data.get("documents") or []
    if not documents:
        raise HTTPException(status_code=404, detail="No document found for that leave request.")

    document = documents[-1]
    download = generate_leave_document_download_url(document["s3Key"])

    return LeaveDocumentDownloadUrlResponse(
        download_url=download["download_url"],
        original_filename=document["originalFilename"],
        expires_in_seconds=download["expires_in_seconds"],
    )
