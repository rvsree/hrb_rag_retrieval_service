"""Manual keyword routing to hrb_lms_mcp's MCP server - ported from
hrb_chatbot_service's ai/rag_pipeline/tools/mcp_tools/__init__.py, using
this repo's own hrb_common.mcp.client + hrb_common.auth.OAuth2ClientCredentials
instead of a hand-rolled MCP session + the original's dedicated oauth_client.

Simplification, flagged: the original's Postgres-backed circuit breaker
(McpRegistryClient pre-flight check / outcome recording) is NOT ported
this phase - every call just attempts the real request directly. A
down/unreachable MCP server still degrades gracefully (falls through to
RAG, same as before), just without the "fail fast on a known outage"
optimization. A follow-up phase can add it back if needed."""

import time

from hrb_common.auth.oauth_client import OAuth2ClientCredentials
from hrb_common.mcp.client import call_mcp_tool
from hrb_common.observability.logger import get_logger
from hrb_common.observability.secrets import read_secret

logger = get_logger("genai.mcp_tools")

LEAVE_BALANCE_KEYWORDS = ("leave balance", "pto balance", "how many days off", "sick balance")
LEAVE_HISTORY_KEYWORDS = ("leave history", "past leave", "leave requests")

_oauth_client: OAuth2ClientCredentials | None = None


def _get_oauth_client() -> OAuth2ClientCredentials:
    global _oauth_client
    if _oauth_client is None:
        _oauth_client = OAuth2ClientCredentials(
            token_url=read_secret("HRB_LMS_MCP_OAUTH_TOKEN_URL", "http://127.0.0.1:8190/oauth/token"),
            client_id=read_secret("HRB_LMS_MCP_OAUTH_CLIENT_ID", ""),
            client_secret=read_secret("HRB_LMS_MCP_OAUTH_CLIENT_SECRET", ""),
        )
    return _oauth_client


def is_leave_balance_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in LEAVE_BALANCE_KEYWORDS)


def is_leave_history_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in LEAVE_HISTORY_KEYWORDS)


async def _call(tool_name: str, arguments: dict) -> dict:
    mcp_url = read_secret("HRB_LMS_MCP_URL", "http://127.0.0.1:8190/mcp")
    access_token = await _get_oauth_client().get_access_token()
    result = await call_mcp_tool(mcp_url, tool_name, arguments, access_token=access_token)
    content = [block["text"] for block in result.get("content", []) if block.get("type") == "text"]
    return {"tool": tool_name, "content": content}


async def get_leave_balance(employee_id: str) -> dict:
    return await _call("get_leave_balance", {"employee_id": employee_id})


async def get_leave_history(employee_id: str) -> dict:
    return await _call("get_leave_history", {"employee_id": employee_id})


async def try_route_to_mcp(query: str, employee_id: str | None) -> dict | None:
    """Returns a pipeline-shaped answer dict if MCP-routable, else None."""
    if not employee_id:
        return None

    if is_leave_balance_query(query):
        tool_name, caller = "get_leave_balance", get_leave_balance
    elif is_leave_history_query(query):
        tool_name, caller = "get_leave_history", get_leave_history
    else:
        return None

    call_started_at = time.perf_counter()
    try:
        mcp_result = await caller(employee_id)
    except Exception as error:
        logger.warning("MCP call to %s failed for %r, falling back to RAG: %s", tool_name, employee_id, error)
        return None
    call_latency_ms = round((time.perf_counter() - call_started_at) * 1000, 1)

    answer_text = "\n".join(mcp_result["content"]) or "No data returned."
    logger.info("Routed query %r to MCP tool %s for employee_id=%s", query, tool_name, employee_id)
    return {
        "query": query,
        "answer": answer_text,
        "model_used": f"mcp:{tool_name}",
        "sources": [],
        "vector_db": "n/a (mcp)",
        "search_strategy": "n/a (mcp)",
        "applied_filter": None,
        "routed_to": tool_name,
        "tool_latency_ms": call_latency_ms,
        "mcp_arguments": {"employee_id": employee_id},
        "mcp_raw_result": mcp_result["content"],
    }
