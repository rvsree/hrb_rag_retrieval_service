"""Document-level ACL - ported from hrb_chatbot_service's
ai/rag_pipeline/helper/access_control.py, adapted to this repo's own
DocumentMetadataStore instead of a combined db_gateway."""

import json

from rag_retrieval.core.stores import DocumentMetadataStore
from rag_retrieval.models import Role

ACL_BYPASS_ROLES = {Role.HR_SUPPORT.value, Role.ADMIN.value}


def _is_denied(document: dict, role: str | None, department: str | None) -> bool:
    restricted_roles_raw = document.get("restricted_roles")
    restricted_roles = json.loads(restricted_roles_raw) if restricted_roles_raw else None
    if restricted_roles and role not in restricted_roles:
        return True

    doc_department = document.get("department")
    if doc_department and (department or "").strip().lower() != doc_department.strip().lower():
        return True

    return False


async def build_acl_filter(metadata_store: DocumentMetadataStore, role: str | None, department: str | None) -> dict | None:
    """Returns {"document_id": {"$nin": [...]}} excluding every document
    this caller is denied, or None when nothing needs excluding."""
    if role in ACL_BYPASS_ROLES:
        return None

    documents = metadata_store.list_documents()
    denied_ids = [
        document["id"]
        for document in documents
        if document.get("is_current", True) and _is_denied(document, role, department)
    ]
    if not denied_ids:
        return None
    return {"document_id": {"$nin": denied_ids}}
