"""LMS Ops Agent - multi-agentic-rag's domain agent over the caller's own
live leave data. Thin wrapper: reuses single-agentic-rag's existing
MCP-backed tools, routed by keyword like genai/mcp_tools.py's own
try_route_to_mcp() already does. Ported from hrb_chatbot_service's
ai/agents/domain_agents/lms_ops_agent.py."""

from rag_retrieval.genai.mcp_tools import is_leave_history_query
from rag_retrieval.single_agentic.tools import get_leave_balance_tool, get_leave_history_tool


async def run(focus: str, employee_id: str | None) -> tuple[str, list[dict]]:
    if not employee_id or not employee_id.strip():
        return "Error: no employee ID available for this request.", []

    if is_leave_history_query(focus):
        return await get_leave_history_tool(employee_id)
    return await get_leave_balance_tool(employee_id)
