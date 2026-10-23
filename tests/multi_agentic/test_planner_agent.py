"""Tests for the Planner's structured-output LLM call - the LLM is faked,
matching every other client test in this project."""

from rag_retrieval.multi_agentic import planner_agent
from rag_retrieval.multi_agentic.planner_agent import PlannerOutput, PlannerTask


class _FakeRawMessage:
    content = ""
    usage_metadata = None


class FakeStructuredLLM:
    def __init__(self, parsed: PlannerOutput | None):
        self.parsed = parsed

    async def ainvoke(self, messages, config=None):
        return {"parsed": self.parsed, "raw": _FakeRawMessage()}


class FakeUnboundLLM:
    def __init__(self, parsed: PlannerOutput | None):
        self.parsed = parsed

    def with_structured_output(self, schema, include_raw=True):
        return FakeStructuredLLM(self.parsed)


async def test_plan_returns_the_parsed_tasks(monkeypatch):
    parsed = PlannerOutput(
        tasks=[PlannerTask(agent="vector_kb_agent", focus="401k policy", reasoning="a policy question")]
    )
    monkeypatch.setattr(planner_agent, "_build_llm", lambda planner_model=None: FakeUnboundLLM(parsed))

    tasks, turn = await planner_agent.plan("what is the 401k policy")

    assert tasks == [{"agent": "vector_kb_agent", "focus": "401k policy", "reasoning": "a policy question"}]
    assert turn["label"] == "planner"


async def test_plan_falls_back_to_vector_kb_agent_when_structured_output_is_empty(monkeypatch):
    monkeypatch.setattr(planner_agent, "_build_llm", lambda planner_model=None: FakeUnboundLLM(None))

    tasks, _turn = await planner_agent.plan("some question")

    assert len(tasks) == 1
    assert tasks[0]["agent"] == "vector_kb_agent"
    assert tasks[0]["focus"] == "some question"


async def test_plan_can_return_multiple_tasks(monkeypatch):
    parsed = PlannerOutput(
        tasks=[
            PlannerTask(agent="vector_kb_agent", focus="401k policy", reasoning="policy part"),
            PlannerTask(agent="lms_ops_agent", focus="my leave balance", reasoning="live data part"),
        ]
    )
    monkeypatch.setattr(planner_agent, "_build_llm", lambda planner_model=None: FakeUnboundLLM(parsed))

    tasks, _turn = await planner_agent.plan("what is the 401k policy and what's my leave balance")

    assert [task["agent"] for task in tasks] == ["vector_kb_agent", "lms_ops_agent"]
