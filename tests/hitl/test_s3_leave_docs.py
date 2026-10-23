from rag_retrieval.hitl import s3_leave_docs


def test_upload_url_key_matches_token_and_filename(monkeypatch):
    monkeypatch.setenv("LEAVE_DOCS_S3_BUCKET", "test-bucket")

    def fake_presign(bucket, key, content_type, expiry_seconds, region):
        return {"upload_url": f"https://example/{bucket}/{key}"}

    monkeypatch.setattr(s3_leave_docs, "_generate_presigned_upload_url", fake_presign)

    result = s3_leave_docs.generate_leave_document_upload_url("tok123", "doc.pdf", "application/pdf")

    assert result["key"] == "tok123/doc.pdf"
    assert "test-bucket" in result["upload_url"]
