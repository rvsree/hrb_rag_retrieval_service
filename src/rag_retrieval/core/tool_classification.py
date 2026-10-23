"""Classifies a tool/agent name by which kind of backend it hits - ported
as-is from hrb_chatbot_service's ai/rag_core/tool_classification.py."""

TOOL_TYPE_BY_NAME = {
    "SearchKnowledgeBase": "vector_db",
    "vector_kb_agent": "vector_db",
    "GetLeaveBalance": "mcp",
    "GetLeaveHistory": "mcp",
    "get_leave_balance": "mcp",
    "get_leave_history": "mcp",
    "lms_ops_agent": "mcp",
    "web_search_agent": "web_search",
    "sql_db_agent": "sql_db",
    "lms_analytics_agent": "sql_db",
}


def classify_tool(tool_name: str) -> str:
    return TOOL_TYPE_BY_NAME.get(tool_name, "other")
