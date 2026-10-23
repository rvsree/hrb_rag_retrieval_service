"""Shared summarize-keyword check - ported as-is from hrb_chatbot_service's
ai/rag_core/response_format.py."""

SUMMARIZE_KEYWORDS = ("summarize", "summary", "in a table", "table format", "tabular")

TABULAR_FORMAT_INSTRUCTION = (
    "If the content is naturally tabular (comparable items, multiple values per "
    "category), format your answer as a markdown table using GFM syntax "
    "(| column | column |) instead of plain prose."
)


def should_use_tabular_format(query: str) -> bool:
    lowered_query = query.lower()
    return any(keyword in lowered_query for keyword in SUMMARIZE_KEYWORDS)
