"""A representative subset of hrb_chatbot_service's own
test_unpaid_leave_agent.py - the eligibility-check (repeat-within-12-months)
logic, the part most worth porting faithfully since it's real business
logic, not just plumbing. Not an exhaustive 1:1 port (the full round-trip
submit-then-resolve graph tests aren't ported this phase) - same
"representative, not exhaustive" scope narrowing Phase 1 already used.

Every hrb_lms_mcp call and LLM call is faked - no real MCP/network call.
The checkpointer is a real in-memory LangGraph MemorySaver, not the real
Postgres-backed one - genuine pause/resume semantics are still exercised."""

from langgraph.checkpoint.memory import MemorySaver

from rag_retrieval.hitl import unpaid_leave_agent


def _patch_checkpointer(monkeypatch):
    saver = MemorySaver()

    async def _fake_run_graph(initial_state_or_command, config):
        graph = unpaid_leave_agent._build_graph().compile(checkpointer=saver)
        return await graph.ainvoke(initial_state_or_command, config=config)

    monkeypatch.setattr(unpaid_leave_agent, "_run_graph", _fake_run_graph)


def _patch_llm(monkeypatch):
    class _FakeResponse:
        def __init__(self, content):
            self.content = content

    class _FakeLLM:
        async def ainvoke(self, messages, config=None):
            return _FakeResponse("Nothing stands out.")

    monkeypatch.setattr(unpaid_leave_agent, "build_agent_llm", lambda: _FakeLLM())


def _patch_review_brief(monkeypatch):
    async def _fake_set_review_brief(hitl_request_context_id, review_brief):
        return {"hitlRequestId": hitl_request_context_id, "reviewBriefSet": True}

    monkeypatch.setattr(unpaid_leave_agent, "set_review_brief", _fake_set_review_brief)


def _patch_history(monkeypatch, requests):
    async def _fake_get_leave_history_parsed(employee_id):
        return {"employeeId": employee_id, "total": len(requests), "requests": requests}

    monkeypatch.setattr(unpaid_leave_agent, "get_leave_history_parsed", _fake_get_leave_history_parsed)


def _patch_submit(monkeypatch, captured_calls):
    async def _fake_submit_unpaid_leave_request(**kwargs):
        captured_calls.append(kwargs)
        return {
            "requestId": 1,
            "employeeId": kwargs["employee_id"],
            "leaveType": "UNPAID",
            "startDate": kwargs["start_date"],
            "endDate": kwargs["end_date"],
            "days": 3.0,
            "status": "PENDING_HITL",
            "isRepeatWithin12Months": kwargs["is_repeat_within_12_months"],
            "hitlRequestId": 7,
            "hitlStatus": "PENDING",
        }

    monkeypatch.setattr(unpaid_leave_agent, "submit_unpaid_leave_request", _fake_submit_unpaid_leave_request)


def _patch_thread_store(monkeypatch):
    saved = {}

    class _FakeThreadStore:
        def save_thread_mapping(self, hitl_request_context_id, agent_thread_id):
            saved[hitl_request_context_id] = agent_thread_id

        def get_thread_id(self, hitl_request_context_id):
            return saved.get(hitl_request_context_id)

    monkeypatch.setattr(unpaid_leave_agent, "_thread_store", _FakeThreadStore())


async def test_no_prior_unpaid_leave_is_not_flagged_as_repeat(monkeypatch):
    _patch_history(monkeypatch, [])
    captured_calls = []
    _patch_submit(monkeypatch, captured_calls)
    _patch_llm(monkeypatch)
    _patch_review_brief(monkeypatch)
    _patch_checkpointer(monkeypatch)
    _patch_thread_store(monkeypatch)

    result = await unpaid_leave_agent.run_unpaid_leave_request(
        employee_id="EMP052", start_date="2026-12-01", end_date="2026-12-03", purpose="Family caregiving"
    )

    assert captured_calls[0]["is_repeat_within_12_months"] is False
    assert result["isRepeatWithin12Months"] is False


async def test_unpaid_leave_within_last_365_days_is_flagged_as_repeat(monkeypatch):
    _patch_history(
        monkeypatch,
        [{"requestId": 1, "leaveType": "UNPAID", "startDate": "2026-06-01", "endDate": "2026-06-05", "days": 5.0, "status": "APPROVED"}],
    )
    captured_calls = []
    _patch_submit(monkeypatch, captured_calls)
    _patch_llm(monkeypatch)
    _patch_review_brief(monkeypatch)
    _patch_checkpointer(monkeypatch)
    _patch_thread_store(monkeypatch)

    result = await unpaid_leave_agent.run_unpaid_leave_request(
        employee_id="EMP052", start_date="2026-12-01", end_date="2026-12-03", purpose="Family caregiving"
    )

    assert captured_calls[0]["is_repeat_within_12_months"] is True
    assert result["isRepeatWithin12Months"] is True


async def test_unpaid_leave_older_than_365_days_is_not_flagged_as_repeat(monkeypatch):
    _patch_history(
        monkeypatch,
        [{"requestId": 1, "leaveType": "UNPAID", "startDate": "2024-01-01", "endDate": "2024-01-05", "days": 5.0, "status": "APPROVED"}],
    )
    captured_calls = []
    _patch_submit(monkeypatch, captured_calls)
    _patch_llm(monkeypatch)
    _patch_review_brief(monkeypatch)
    _patch_checkpointer(monkeypatch)
    _patch_thread_store(monkeypatch)

    await unpaid_leave_agent.run_unpaid_leave_request(
        employee_id="EMP052", start_date="2026-12-01", end_date="2026-12-03", purpose="Family caregiving"
    )

    assert captured_calls[0]["is_repeat_within_12_months"] is False


async def test_repeat_check_is_scoped_to_the_same_leave_type_only(monkeypatch):
    _patch_history(
        monkeypatch,
        [{"requestId": 1, "leaveType": "CRITICAL_CAREGIVER", "startDate": "2026-06-01", "endDate": "2026-06-05", "days": 5.0, "status": "APPROVED"}],
    )
    captured_calls = []
    _patch_submit(monkeypatch, captured_calls)
    _patch_llm(monkeypatch)
    _patch_review_brief(monkeypatch)
    _patch_checkpointer(monkeypatch)
    _patch_thread_store(monkeypatch)

    await unpaid_leave_agent.run_unpaid_leave_request(
        employee_id="EMP052", start_date="2026-12-01", end_date="2026-12-03", purpose="Family caregiving", leave_type_code="UNPAID"
    )

    assert captured_calls[0]["is_repeat_within_12_months"] is False


async def test_resume_with_an_unknown_hitl_request_context_id_raises(monkeypatch):
    _patch_thread_store(monkeypatch)

    try:
        await unpaid_leave_agent.resume_unpaid_leave_decision(
            hitl_request_context_id=999, decision="APPROVED", response_text="ok"
        )
        assert False, "expected ValueError"
    except ValueError as error:
        assert "999" in str(error)
