"""Presigned upload/download URLs for leave-request documents - ported from
hrb_chatbot_service's common/clients/storage_client/s3_upload_client.py,
narrowed to the 2 leave-document functions this phase needs. Reads
LEAVE_DOCS_S3_BUCKET, never S3_UPLOAD_BUCKET - that bucket has a
whole-bucket S3 event notification feeding the KB ingestion pipeline, so a
leave document dropped there would silently get chunked/embedded into the
shared RAG vector index. Built on hrb_common.storage's generic (bucket,
key) functions, same pattern rag_ingestion_service's own s3_upload.py uses."""

from hrb_common.observability.secrets import read_secret
from hrb_common.storage.s3_client import delete_object
from hrb_common.storage.s3_client import (
    generate_presigned_download_url as _generate_presigned_download_url,
)
from hrb_common.storage.s3_client import (
    generate_presigned_upload_url as _generate_presigned_upload_url,
)

DEFAULT_REGION = "us-east-1"
DEFAULT_EXPIRY_SECONDS = 300


def generate_leave_document_upload_url(leave_request_token: str, filename: str, content_type: str) -> dict:
    bucket = read_secret("LEAVE_DOCS_S3_BUCKET", "")
    region = read_secret("AWS_REGION", DEFAULT_REGION)
    expiry_seconds = int(read_secret("S3_PRESIGNED_URL_EXPIRY_SECONDS", str(DEFAULT_EXPIRY_SECONDS)))
    key = f"{leave_request_token}/{filename}"

    result = _generate_presigned_upload_url(bucket, key, content_type, expiry_seconds=expiry_seconds, region=region)
    return {"upload_url": result["upload_url"], "key": key, "expires_in_seconds": expiry_seconds}


def generate_leave_document_download_url(s3_key: str) -> dict:
    bucket = read_secret("LEAVE_DOCS_S3_BUCKET", "")
    region = read_secret("AWS_REGION", DEFAULT_REGION)
    expiry_seconds = int(read_secret("S3_PRESIGNED_URL_EXPIRY_SECONDS", str(DEFAULT_EXPIRY_SECONDS)))

    result = _generate_presigned_download_url(bucket, s3_key, expiry_seconds=expiry_seconds, region=region)
    return {"download_url": result["download_url"], "expires_in_seconds": expiry_seconds}


def delete_leave_document_object(s3_key: str) -> None:
    """Best-effort - deleting an already-missing key is a no-op, not an error."""
    bucket = read_secret("LEAVE_DOCS_S3_BUCKET", "")
    region = read_secret("AWS_REGION", DEFAULT_REGION)
    delete_object(bucket, s3_key, region=region)
