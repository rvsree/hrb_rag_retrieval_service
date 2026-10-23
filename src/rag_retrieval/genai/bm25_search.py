"""Keyword (BM25) and Hybrid (BM25 + vector, Reciprocal Rank Fusion) search
strategies - ported as-is from hrb_chatbot_service's
ai/rag_pipeline/query_retrieval/bm25_search.py. Architecture unchanged -
the Qdrant/persisted-sparse-index question is a separate, explicitly
deferred decision in hrb_chatbot_service itself; this just ports what
exists today."""

from rank_bm25 import BM25Okapi

from hrb_common.observability.logger import get_logger
from rag_retrieval.gateway import COLLECTION_NAME, get_raw_vector_client

logger = get_logger("genai.bm25_search")

RRF_K = 60


def _matches_filter_clause(metadata: dict, clause: dict | None) -> bool:
    if not clause:
        return True
    if "$and" in clause:
        return all(_matches_filter_clause(metadata, sub_clause) for sub_clause in clause["$and"])
    for key, condition in clause.items():
        actual = metadata.get(key)
        if isinstance(condition, dict):
            if "$ne" in condition and actual == condition["$ne"]:
                return False
            if "$nin" in condition and actual in condition["$nin"]:
                return False
        elif actual != condition:
            return False
    return True


def _matches_metadata_filter(metadata: dict, extra_filter: dict | None) -> bool:
    if metadata.get("is_current") is False:
        return False
    return _matches_filter_clause(metadata, extra_filter)


def search_keyword(query: str, top_k: int = 5, vector_db: str | None = None, extra_filter: dict | None = None) -> list[dict]:
    """Plain BM25 lexical ranking over every current chunk."""
    vector_store_client = get_raw_vector_client(vector_db)
    all_chunks = vector_store_client.get_all_chunks(COLLECTION_NAME)
    current_chunks = [chunk for chunk in all_chunks if _matches_metadata_filter(chunk["metadata"], extra_filter)]

    if not current_chunks:
        return []

    tokenized_corpus = [chunk["text"].lower().split() for chunk in current_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(zip(current_chunks, scores), key=lambda pair: pair[1], reverse=True)
    results = []
    for chunk, score in ranked:
        if score <= 0 or len(results) >= top_k:
            continue
        results.append(
            {
                "document_id": chunk["metadata"].get("document_id"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "text": chunk["text"],
                "score": float(score),
            }
        )
    return results


def _chunk_key(chunk: dict) -> tuple:
    return (chunk["document_id"], chunk["chunk_index"])


def _reciprocal_rank_fusion(labeled_ranked_lists: dict[str, list[dict]], top_k: int) -> list[dict]:
    rrf_scores: dict[tuple, float] = {}
    chunks_by_key: dict[tuple, dict] = {}
    matched_by: dict[tuple, list[str]] = {}

    for label, ranked_list in labeled_ranked_lists.items():
        for position, chunk in enumerate(ranked_list):
            key = _chunk_key(chunk)
            rank = position + 1
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunks_by_key.setdefault(key, chunk)
            matched_by.setdefault(key, []).append(label)

    ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)[:top_k]
    results = []
    for key in ranked_keys:
        fused_chunk = dict(chunks_by_key[key])
        fused_chunk["score"] = rrf_scores[key]
        fused_chunk["matched_by"] = matched_by[key]
        results.append(fused_chunk)
    return results


def search_hybrid(query: str, top_k: int = 5, vector_db: str | None = None, extra_filter: dict | None = None) -> list[dict]:
    """BM25 (lexical) + similarity (semantic), merged with Reciprocal Rank Fusion."""
    from rag_retrieval.genai.retriever import search_similarity

    fetch_k = top_k * 3
    keyword_results = search_keyword(query, top_k=fetch_k, vector_db=vector_db, extra_filter=extra_filter)
    vector_results = search_similarity(query, top_k=fetch_k, vector_db=vector_db, extra_filter=extra_filter)
    return _reciprocal_rank_fusion({"sparse": keyword_results, "dense": vector_results}, top_k)
