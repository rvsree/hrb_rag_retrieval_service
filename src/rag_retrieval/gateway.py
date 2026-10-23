"""Resolves which concrete backend to use for this repo's own retrieval code.

hrb_chatbot_service reaches these through its own combined db_gateway/client_gateway
singletons. This repo has no such combined gateway (hrb-common-libs deliberately
keeps db/llm/vectorstore as separate extras) - this module is the one place that
picks a provider by name, mirroring rag_ingestion_service's own gateway.py pattern."""

from langchain_openai import ChatOpenAI

from hrb_common.cache.connection import get_redis_client
from hrb_common.db.connection import PostgresConnectionConfig, get_connection
from hrb_common.observability.secrets import read_secret
from hrb_common.vectorstore.chroma_client import ChromaDBClient
from hrb_common.vectorstore.langchain_vector_store import COLLECTION_NAME, get_vector_store
from hrb_common.vectorstore.pinecone_client import PineconeClient

__all__ = [
    "COLLECTION_NAME",
    "get_vector_store",
    "get_raw_vector_client",
    "get_postgres_connection",
    "get_redis",
    "build_chat_llm",
]

_raw_vector_clients: dict[str, object] = {}


def get_raw_vector_client(provider: str | None = None):
    """The RAW hrb_common client (ChromaDBClient/PineconeClient), not
    LangChain-wrapped - needed for get_all_chunks(), which only this
    project's own client classes expose, not LangChain's VectorStore
    interface. Cached per-provider, same as get_vector_store()'s own
    underlying client cache in langchain_vector_store.py."""
    resolved_provider = read_secret("ACTIVE_VECTOR_DB", "chromadb", passed_in_value=provider)
    if resolved_provider not in _raw_vector_clients:
        if resolved_provider == "pinecone":
            _raw_vector_clients[resolved_provider] = PineconeClient()
        else:
            _raw_vector_clients[resolved_provider] = ChromaDBClient()
    return _raw_vector_clients[resolved_provider]


def get_postgres_connection():
    return get_connection(PostgresConnectionConfig.from_env())


def get_redis():
    return get_redis_client()


def build_chat_llm(provider: str | None = None, model_name: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """A real LangChain BaseChatModel, constructed explicitly with the API key
    from read_secret() - not left to LangChain's own default os.environ
    resolution, which bypasses the Secrets Manager fallback (same real bug
    class rag_ingestion_service's indexing.py found and fixed). OpenAI only -
    this repo has no multi-provider LangChain chat model wrapper equivalent
    to hrb_chatbot_service's own GatewayChatModel; "openai" is the only
    provider genai-rag's multi_query/self_query call sites actually use in
    practice (llm_provider defaults to "openai"), so that gap is accepted
    here rather than built out, flagged in the delivery note."""
    api_key = read_secret("OPENAI_API_KEY", "")
    base_url = (read_secret("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/")
    resolved_model = read_secret("OPENAI_CHAT_MODEL", "gpt-4.1-mini", passed_in_value=model_name)
    return ChatOpenAI(model=resolved_model, api_key=api_key, base_url=base_url, temperature=temperature)
