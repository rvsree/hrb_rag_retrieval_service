"""Exact KNN search strategy - ported as-is from hrb_chatbot_service's
ai/rag_pipeline/query_retrieval/exact_knn_search.py."""

import math

from hrb_common.observability.logger import get_logger
from rag_retrieval.gateway import COLLECTION_NAME, get_raw_vector_client, get_vector_store
from rag_retrieval.genai.bm25_search import _matches_metadata_filter

logger = get_logger("genai.exact_knn_search")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def search_knn(query: str, top_k: int = 5, vector_db: str | None = None, extra_filter: dict | None = None) -> list[dict]:
    """Exact (not approximate) nearest-neighbor search - brute-force cosine
    similarity against every current chunk's own stored embedding."""
    store, provider = get_vector_store(vector_db)
    query_embedding = store.embeddings.embed_query(query)

    vector_store_client = get_raw_vector_client(vector_db)
    all_chunks = vector_store_client.get_all_chunks(COLLECTION_NAME)
    current_chunks = [
        chunk
        for chunk in all_chunks
        if _matches_metadata_filter(chunk["metadata"], extra_filter) and chunk.get("embedding")
    ]

    scored = [(chunk, _cosine_similarity(query_embedding, chunk["embedding"])) for chunk in current_chunks]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        results.append(
            {
                "document_id": chunk["metadata"].get("document_id"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "text": chunk["text"],
                "score": score,
            }
        )
    logger.info("Exact KNN search (%s): %d chunk(s) scored, top %d returned", provider, len(current_chunks), len(results))
    return results
