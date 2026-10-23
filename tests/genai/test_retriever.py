from langchain_core.documents import Document

from rag_retrieval.genai import retriever


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def similarity_search_with_score(self, query, k, filter):
        return self._results


def test_combine_with_current_only_wraps_with_and_when_extra_filter_given():
    combined = retriever._combine_with_current_only({"department": "HR"})
    assert combined == {"$and": [retriever.CURRENT_CHUNKS_ONLY, {"department": "HR"}]}


def test_combine_with_current_only_returns_bare_filter_when_none_given():
    assert retriever._combine_with_current_only(None) == retriever.CURRENT_CHUNKS_ONLY


def test_document_to_chunk_maps_metadata():
    doc = Document(page_content="hello", metadata={"document_id": "doc-1", "chunk_index": 2})
    chunk = retriever._document_to_chunk(doc, 0.9)
    assert chunk == {"document_id": "doc-1", "chunk_index": 2, "text": "hello", "score": 0.9}


def test_search_similarity_drops_chunks_below_relevance_bar(monkeypatch):
    doc_good = Document(page_content="good", metadata={"document_id": "a", "chunk_index": 0})
    doc_bad = Document(page_content="bad", metadata={"document_id": "b", "chunk_index": 0})
    fake_store = FakeVectorStore([(doc_good, 0.5), (doc_bad, 5.0)])  # chroma: lower is better
    monkeypatch.setattr(retriever, "get_vector_store", lambda vector_db: (fake_store, "chromadb"))

    results = retriever.search_similarity("query", top_k=5)

    assert len(results) == 1
    assert results[0]["document_id"] == "a"


def test_combine_extra_filters_ands_both_when_both_present():
    combined = retriever._combine_extra_filters({"a": 1}, {"b": 2})
    assert combined == {"$and": [{"a": 1}, {"b": 2}]}


def test_combine_extra_filters_returns_none_when_neither_present():
    assert retriever._combine_extra_filters(None, None) is None
