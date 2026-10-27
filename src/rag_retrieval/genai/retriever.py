"""Searches the vector store for chunks matching a query - similarity/MMR,
plus LangChain's own retrievers layered on top. Ported from
hrb_chatbot_service's ai/rag_pipeline/query_retrieval/retriever.py.

Adaptation: GatewayChatModel (hrb_chatbot_service's own multi-provider
LangChain chat model) has no equivalent in hrb-common-libs - MultiQuery/
Self-Query use this repo's own gateway.build_chat_llm() (OpenAI-only,
constructed explicitly with the real API key, not left to LangChain's
default os.environ resolution - the real bug class rag_ingestion_service's
indexing.py already found and fixed)."""

import difflib
import re

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain_core.documents import Document

from hrb_common.observability.logger import get_logger
from hrb_common.observability.secrets import read_secret
from rag_retrieval.core.stores import DocumentMetadataStore
from rag_retrieval.gateway import build_chat_llm, get_vector_store
from rag_retrieval.genai.access_control import build_acl_filter

logger = get_logger("genai.retriever")

CURRENT_CHUNKS_ONLY = {"is_current": {"$ne": False}}

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="doc_category",
        description="The kind of document: one of policy, regulatory, investment, benefits, other.",
        type="string",
    ),
    AttributeInfo(name="department", description="Which company department this document belongs to.", type="string"),
    AttributeInfo(
        name="doc_description",
        description=(
            "The specific topic this document covers, in the document's own terms, e.g. '401k', "
            "'health benefits', 'leave policy'."
        ),
        type="string",
    ),
]
DOCUMENT_CONTENTS_DESCRIPTION = (
    "Sections of Amazing Bank HR benefits policy documents - 401k, healthcare, paid/unpaid time off, "
    "tuition assistance, and similar."
)

_STRUCTURED_QUERY_TRANSLATORS = {"chromadb": ChromaTranslator, "pinecone": PineconeTranslator}

MAX_CHROMA_DISTANCE = float(read_secret("RAG_MAX_CHROMA_DISTANCE", "1.1"))
MIN_PINECONE_SCORE = float(read_secret("RAG_MIN_PINECONE_SCORE", "0.5"))

_metadata_store = DocumentMetadataStore()


def _meets_relevance_bar(score: float, provider_name: str) -> bool:
    if provider_name == "pinecone":
        return score >= MIN_PINECONE_SCORE
    return score <= MAX_CHROMA_DISTANCE


def _document_to_chunk(document: Document, score: float | None) -> dict:
    metadata = document.metadata
    return {
        "document_id": metadata.get("document_id"),
        "chunk_index": metadata.get("chunk_index"),
        "text": document.page_content,
        "score": score,
    }


def _combine_with_current_only(extra_filter: dict | None) -> dict:
    if not extra_filter:
        return CURRENT_CHUNKS_ONLY
    return {"$and": [CURRENT_CHUNKS_ONLY, extra_filter]}


def search_similarity(
    query: str, top_k: int = 5, vector_db: str | None = None, extra_filter: dict | None = None
) -> list[dict]:
    store, provider = get_vector_store(vector_db)
    results = store.similarity_search_with_score(query, k=top_k, filter=_combine_with_current_only(extra_filter))

    chunks = []
    for document, score in results:
        if _meets_relevance_bar(score, provider):
            chunks.append(_document_to_chunk(document, score))
    return chunks


def search_mmr(
    query: str,
    top_k: int = 5,
    vector_db: str | None = None,
    fetch_k: int | None = None,
    extra_filter: dict | None = None,
    lambda_mult: float | None = None,
) -> list[dict]:
    store, provider = get_vector_store(vector_db)
    documents = store.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=fetch_k or top_k * 3,
        filter=_combine_with_current_only(extra_filter),
        lambda_mult=lambda_mult if lambda_mult is not None else 0.5,
    )
    return [_document_to_chunk(document, None) for document in documents]


SEARCH_STRATEGIES = {
    "similarity": search_similarity,
    "mmr": search_mmr,
}

# Imported down here, after SEARCH_STRATEGIES exists, same circular-import
# avoidance reason as the original.
from rag_retrieval.genai.bm25_search import search_hybrid, search_keyword  # noqa: E402
from rag_retrieval.genai.exact_knn_search import search_knn  # noqa: E402

SEARCH_STRATEGIES["keyword"] = search_keyword
SEARCH_STRATEGIES["hybrid"] = search_hybrid
SEARCH_STRATEGIES["knn"] = search_knn


def search_multi_query(
    query: str,
    top_k: int = 5,
    vector_db: str | None = None,
    search_strategy: str | None = None,
    llm_provider: str = "openai",
    extra_filter: dict | None = None,
) -> list[dict]:
    store, _ = get_vector_store(vector_db)
    resolved_strategy = search_strategy or "similarity"
    search_kwargs = {"k": top_k, "filter": _combine_with_current_only(extra_filter)}
    if resolved_strategy == "mmr":
        search_kwargs["fetch_k"] = top_k * 3
    base_retriever = store.as_retriever(search_type=resolved_strategy, search_kwargs=search_kwargs)

    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=build_chat_llm(llm_provider))
    documents = multi_query_retriever.invoke(query)
    return [_document_to_chunk(document, None) for document in documents]


DOC_DESCRIPTION_MATCH_CUTOFF = 0.6


def _normalize_for_matching(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _resolve_doc_description_value(value: str, real_descriptions: list[str]) -> str:
    normalized_value = _normalize_for_matching(value)
    if normalized_value:
        for real_value in real_descriptions:
            if normalized_value in _normalize_for_matching(real_value):
                return real_value

    matches = difflib.get_close_matches(value, real_descriptions, n=1, cutoff=DOC_DESCRIPTION_MATCH_CUTOFF)
    return matches[0] if matches else value


def _rewrite_doc_description_filter(filter_value, real_descriptions: list[str]):
    if isinstance(filter_value, dict):
        if "doc_description" in filter_value and isinstance(filter_value["doc_description"], dict):
            eq_clause = filter_value["doc_description"]
            if "$eq" in eq_clause:
                return {
                    **filter_value,
                    "doc_description": {**eq_clause, "$eq": _resolve_doc_description_value(eq_clause["$eq"], real_descriptions)},
                }
        return {key: _rewrite_doc_description_filter(value, real_descriptions) for key, value in filter_value.items()}
    if isinstance(filter_value, list):
        return [_rewrite_doc_description_filter(item, real_descriptions) for item in filter_value]
    return filter_value


def _parse_self_query_filter(store, provider: str, query: str, llm_provider: str) -> dict | None:
    translator = _STRUCTURED_QUERY_TRANSLATORS[provider]()
    self_query_retriever = SelfQueryRetriever.from_llm(
        build_chat_llm(llm_provider),
        store,
        DOCUMENT_CONTENTS_DESCRIPTION,
        METADATA_FIELD_INFO,
        structured_query_translator=translator,
    )
    structured_query = self_query_retriever.query_constructor.invoke({"query": query})
    _, search_kwargs = self_query_retriever._prepare_query(query, structured_query)
    parsed_filter = search_kwargs.get("filter")
    if parsed_filter is None:
        return None

    if "doc_description" in str(parsed_filter):
        documents = _metadata_store.list_documents()
        real_descriptions = [doc["doc_description"] for doc in documents if doc.get("doc_description")]
        if real_descriptions:
            parsed_filter = _rewrite_doc_description_filter(parsed_filter, real_descriptions)
    return parsed_filter


def _combine_extra_filters(self_query_filter: dict | None, acl_filter: dict | None) -> dict | None:
    filters = [f for f in (self_query_filter, acl_filter) if f]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


async def retrieve_chunks(
    query: str,
    top_k: int = 5,
    vector_db: str | None = None,
    search_strategy: str | None = None,
    use_multi_query: bool = False,
    use_self_query: bool = False,
    llm_provider: str | None = None,
    lambda_mult: float | None = None,
    role: str | None = None,
    department: str | None = None,
) -> tuple[list[dict], dict | None]:
    """Returns (chunks, applied_filter)."""
    resolved_strategy = search_strategy or "similarity"
    resolved_llm_provider = llm_provider or "openai"
    logger.info(
        "search: strategy=%s (%s), multi_query=%s, self_query=%s, llm_provider=%s",
        resolved_strategy,
        "explicit" if search_strategy else "default",
        use_multi_query,
        use_self_query,
        resolved_llm_provider,
    )
    if resolved_strategy not in SEARCH_STRATEGIES:
        raise ValueError(f"Unknown search_strategy {resolved_strategy!r} - choose one of {list(SEARCH_STRATEGIES)}")

    store, provider = get_vector_store(vector_db)

    applied_filter: dict | None = None
    if use_self_query:
        try:
            applied_filter = _parse_self_query_filter(store, provider, query, resolved_llm_provider)
        except Exception as error:
            logger.warning(
                "Self-Query filter parsing failed for %r - falling back to an unfiltered search: %s: %s",
                query, type(error).__name__, error,
            )
            applied_filter = None

    try:
        acl_filter = await build_acl_filter(_metadata_store, role, department)
    except Exception as error:
        logger.warning("ACL filter resolution failed - falling back to an unfiltered search: %s: %s", type(error).__name__, error)
        acl_filter = None
    combined_filter = _combine_extra_filters(applied_filter, acl_filter)

    if use_multi_query:
        chunks = search_multi_query(
            query, top_k=top_k, vector_db=vector_db, search_strategy=resolved_strategy,
            llm_provider=resolved_llm_provider, extra_filter=combined_filter,
        )
    else:
        search_function = SEARCH_STRATEGIES[resolved_strategy]
        search_call_kwargs = {"top_k": top_k, "vector_db": vector_db, "extra_filter": combined_filter}
        if resolved_strategy == "mmr" and lambda_mult is not None:
            search_call_kwargs["lambda_mult"] = lambda_mult
        chunks = search_function(query, **search_call_kwargs)

    _attach_filenames(chunks)

    logger.info("Retrieved %d chunk(s) via %s, applied_filter=%s, acl_filter=%s", len(chunks), resolved_strategy, applied_filter, acl_filter)
    return chunks, applied_filter


def _attach_filenames(chunks: list[dict]) -> None:
    filenames_by_document_id: dict[str, str] = {}
    for chunk in chunks:
        document_id = chunk["document_id"]
        if document_id not in filenames_by_document_id:
            document = _metadata_store.get_document(document_id)
            filenames_by_document_id[document_id] = document["filename"] if document else "unknown"
        chunk["filename"] = filenames_by_document_id[document_id]
