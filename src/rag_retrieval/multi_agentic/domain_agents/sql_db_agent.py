"""SQL DB Agent - multi-agentic-rag's domain agent stub. No structured HR
operational database exists yet - reports that honestly rather than
inventing a result. Ported as-is from hrb_chatbot_service's
ai/agents/domain_agents/sql_db_agent.py."""

NOT_AVAILABLE_MESSAGE = (
    "The SQL DB Agent isn't available yet - there's no structured HR operational "
    "database wired up for this project. This part of the question couldn't be answered."
)


async def run(focus: str) -> str:
    return NOT_AVAILABLE_MESSAGE
