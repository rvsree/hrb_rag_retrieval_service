"""Tests for genai/reviewer.py::generate_follow_ups() - added Phase 3 for
multi-agentic-rag's own reviewer_node. review() (Phase 1) has no dedicated
test yet - a pre-existing gap, not backfilled here, out of this phase's scope."""

from langchain_core.messages import AIMessage

from rag_retrieval.genai import reviewer


class FakeLLM:
    def __init__(self, response_text: str | None = None, raises: bool = False):
        self.response_text = response_text
        self.raises = raises

    async def ainvoke(self, messages, config=None):
        if self.raises:
            raise RuntimeError("LLM call failed")
        return AIMessage(content=self.response_text)


async def test_generate_follow_ups_splits_lines(monkeypatch):
    monkeypatch.setattr(reviewer, "build_agent_llm", lambda: FakeLLM("What about dental coverage?\nHow do I enroll?"))

    follow_ups = await reviewer.generate_follow_ups("What is the 401k match?", "JPMC matches up to 5%.")

    assert follow_ups == ["What about dental coverage?", "How do I enroll?"]


async def test_generate_follow_ups_returns_empty_list_on_failure(monkeypatch):
    monkeypatch.setattr(reviewer, "build_agent_llm", lambda: FakeLLM(raises=True))

    follow_ups = await reviewer.generate_follow_ups("question", "answer")

    assert follow_ups == []
