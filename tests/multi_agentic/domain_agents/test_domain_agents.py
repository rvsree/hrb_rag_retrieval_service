"""Each domain agent is a thin wrapper - tests fake the one underlying tool
each wraps, matching this project's "no real API/network call" convention."""

from rag_retrieval.multi_agentic.domain_agents import (
    lms_analytics_agent,
    lms_ops_agent,
    sql_db_agent,
    vector_kb_agent,
    web_search_agent,
)


async def test_sql_db_agent_reports_honestly_unavailable():
    result = await sql_db_agent.run("some structured query")
    assert "isn't available yet" in result


async def test_lms_analytics_agent_reports_honestly_unavailable():
    result = await lms_analytics_agent.run("aggregate leave stats")
    assert "isn't available yet" in result


async def test_vector_kb_agent_delegates_to_search_knowledge_base(monkeypatch):
    async def _fake_search_knowledge_base(query, search_strategy=None, role=None, department=None):
        return "Found relevant knowledge base content:\n\n--- Source 1 (policy.pdf) ---\nPTO details.\n\n", [
            {"document_id": "doc1", "chunk_index": 0, "filename": "policy.pdf", "text": "PTO details.", "score": 0.9}
        ]

    monkeypatch.setattr(vector_kb_agent, "search_knowledge_base", _fake_search_knowledge_base)

    result, chunks = await vector_kb_agent.run("PTO policy")

    assert "PTO details" in result
    assert len(chunks) == 1


async def test_lms_ops_agent_requires_an_employee_id():
    result, chunks = await lms_ops_agent.run("what is my balance", None)
    assert result.startswith("Error:")
    assert chunks == []


async def test_lms_ops_agent_routes_leave_history_by_keyword(monkeypatch):
    async def _fake_history_tool(employee_id):
        return "history text", []

    async def _fake_balance_tool(employee_id):
        raise AssertionError("should not call balance tool for a history question")

    monkeypatch.setattr(lms_ops_agent, "get_leave_history_tool", _fake_history_tool)
    monkeypatch.setattr(lms_ops_agent, "get_leave_balance_tool", _fake_balance_tool)
    monkeypatch.setattr(lms_ops_agent, "is_leave_history_query", lambda focus: True)

    result, chunks = await lms_ops_agent.run("what leave have I taken", "EMP1")

    assert result == "history text"


async def test_web_search_agent_reports_the_error_as_text(monkeypatch):
    class _FakeTavily:
        def search(self, query):
            return {"results": [], "answer": "", "error": "timeout"}

    monkeypatch.setattr(web_search_agent, "_tavily_client", _FakeTavily())

    result = await web_search_agent.run("current mileage rate")

    assert "web search failed" in result


async def test_web_search_agent_formats_results(monkeypatch):
    class _FakeTavily:
        def search(self, query):
            return {
                "results": [{"title": "IRS Rate", "content": "67 cents per mile", "url": "https://example.com"}],
                "answer": "The current rate is 67 cents per mile.",
            }

    monkeypatch.setattr(web_search_agent, "_tavily_client", _FakeTavily())

    result = await web_search_agent.run("current mileage rate")

    assert "67 cents per mile" in result
    assert "IRS Rate" in result
