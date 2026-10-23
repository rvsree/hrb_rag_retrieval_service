"""Vector KB Agent - multi-agentic-rag's domain agent over the policy
knowledge base. Thin wrapper, no new logic: reuses single-agentic-rag's
existing search tool as-is. Ported from hrb_chatbot_service's
ai/agents/domain_agents/vector_kb_agent.py."""

from rag_retrieval.single_agentic.tools import search_knowledge_base


async def run(
    focus: str, search_strategy: str | None = None, role: str | None = None, department: str | None = None
) -> tuple[str, list[dict]]:
    return await search_knowledge_base(focus, search_strategy=search_strategy, role=role, department=department)
