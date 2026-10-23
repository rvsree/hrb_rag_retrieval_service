from rag_retrieval.genai.bm25_search import _matches_filter_clause, _matches_metadata_filter, _reciprocal_rank_fusion


def test_matches_filter_clause_handles_and_and_nin():
    clause = {"$and": [{"department": "HR"}, {"document_id": {"$nin": ["doc-1"]}}]}
    assert _matches_filter_clause({"department": "HR", "document_id": "doc-2"}, clause) is True
    assert _matches_filter_clause({"department": "HR", "document_id": "doc-1"}, clause) is False


def test_matches_metadata_filter_excludes_non_current_chunks():
    assert _matches_metadata_filter({"is_current": False}, None) is False
    assert _matches_metadata_filter({"is_current": True}, None) is True


def test_reciprocal_rank_fusion_favors_chunks_in_both_lists():
    sparse = [
        {"document_id": "a", "chunk_index": 0, "text": "x"},
        {"document_id": "b", "chunk_index": 0, "text": "y"},
    ]
    dense = [
        {"document_id": "a", "chunk_index": 0, "text": "x"},
        {"document_id": "c", "chunk_index": 0, "text": "z"},
    ]
    fused = _reciprocal_rank_fusion({"sparse": sparse, "dense": dense}, top_k=3)
    assert fused[0]["document_id"] == "a"
    assert fused[0]["matched_by"] == ["sparse", "dense"]
