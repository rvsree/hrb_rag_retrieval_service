"""Web Search Agent - multi-agentic-rag's domain agent for external/current
information the internal KB and HR systems can't answer. Thin wrapper over
hrb-common-libs' own Tavily client - never raises, same convention as every
other domain agent. Ported from hrb_chatbot_service's
ai/agents/domain_agents/web_search_agent.py."""

from hrb_common.web.tavily_client import TavilyClient

_tavily_client = TavilyClient()


async def run(focus: str) -> str:
    result = _tavily_client.search(focus)

    if result.get("error"):
        return f"Error: web search failed ({result['error']})."

    if not result["results"] and not result["answer"]:
        return "No relevant web search results found."

    output = ""
    if result["answer"]:
        output += f"{result['answer']}\n\n"

    for i, item in enumerate(result["results"], 1):
        title = item.get("title", "")
        snippet = item.get("content", "")
        url = item.get("url", "")
        output += f"--- Web Source {i} ({title}) ---\n{snippet}\n({url})\n\n"

    return output.strip()
