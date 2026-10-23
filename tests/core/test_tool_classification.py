from rag_retrieval.core.tool_classification import classify_tool


def test_classifies_known_tool():
    assert classify_tool("GetLeaveBalance") == "mcp"
    assert classify_tool("SearchKnowledgeBase") == "vector_db"


def test_unknown_tool_classifies_as_other():
    assert classify_tool("SomethingNew") == "other"
