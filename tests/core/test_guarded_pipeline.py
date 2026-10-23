from rag_retrieval.core import guarded_pipeline


async def _fake_check_input(query, guardrails_model=None):
    return query, {"status": "pass", "rail": None}


async def _fake_check_output(query, answer, guardrails_model=None):
    return answer, {"status": "pass", "rail": None, "model_used": "gpt-4.1-nano"}


async def _fake_generate(checked_query, chat_history):
    return {"answer": f"answer to {checked_query}"}


async def test_run_guarded_pipeline_happy_path(monkeypatch):
    monkeypatch.setattr(guarded_pipeline, "check_input", _fake_check_input)
    monkeypatch.setattr(guarded_pipeline, "check_output", _fake_check_output)

    async def fake_load_history(conversation_id):
        return []

    async def fake_save_turn(conversation_id, employee_id, human_content, ai_content):
        return None

    monkeypatch.setattr(guarded_pipeline.conversation_memory, "load_history", fake_load_history)
    monkeypatch.setattr(guarded_pipeline.conversation_memory, "save_turn", fake_save_turn)

    result = await guarded_pipeline.run_guarded_pipeline(
        "what is my PTO balance", "EMP1", False, None, _fake_generate
    )

    assert result["answer"] == "answer to what is my PTO balance"
    assert result["guardrails"]["input"]["status"] == "pass"
    assert result["guardrails_model_used"] == "gpt-4.1-nano"


async def test_run_guarded_pipeline_uses_pre_checked_query_when_given(monkeypatch):
    called = {"check_input": False}

    async def _should_not_be_called(*args, **kwargs):
        called["check_input"] = True
        return "unused", {"status": "pass"}

    monkeypatch.setattr(guarded_pipeline, "check_input", _should_not_be_called)
    monkeypatch.setattr(guarded_pipeline, "check_output", _fake_check_output)

    result = await guarded_pipeline.run_guarded_pipeline(
        "raw query", "EMP1", False, None, _fake_generate,
        pre_checked_query="already checked query", pre_checked_input_status={"status": "pass", "rail": None},
    )

    assert called["check_input"] is False
    assert result["answer"] == "answer to already checked query"
