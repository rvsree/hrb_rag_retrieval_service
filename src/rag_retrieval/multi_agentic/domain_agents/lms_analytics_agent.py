"""LMS Analytics Agent - multi-agentic-rag's domain agent stub. No historical
analytics dataset exists yet - reports that honestly rather than inventing
a result. Ported as-is from hrb_chatbot_service's
ai/agents/domain_agents/lms_analytics_agent.py."""

NOT_AVAILABLE_MESSAGE = (
    "The LMS Analytics Agent isn't available yet - there's no historical analytics "
    "dataset wired up for this project. This part of the question couldn't be answered."
)


async def run(focus: str) -> str:
    return NOT_AVAILABLE_MESSAGE
