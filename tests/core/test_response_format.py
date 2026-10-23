from rag_retrieval.core.response_format import should_use_tabular_format


def test_summarize_keyword_triggers_tabular_format():
    assert should_use_tabular_format("summarize my benefits in a table") is True


def test_plain_question_does_not_trigger_tabular_format():
    assert should_use_tabular_format("what is my PTO balance") is False
