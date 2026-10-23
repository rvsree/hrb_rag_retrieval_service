"""Assembles retrieved material into one prompt-ready string - ported as-is
from hrb_chatbot_service's ai/search_pre_processing/context_builder.py."""


def build_context_from_chunks(chunks: list[dict]) -> str:
    blocks = [f"[{chunk['filename']}, chunk {chunk['chunk_index']}]\n{chunk['text']}" for chunk in chunks]
    return "\n\n".join(blocks)


def build_context_from_agent_results(agent_results: list[dict]) -> str:
    """multi-agentic-rag's own context shape - Phase 3's concern, ported
    here for completeness since it lives in the same original file."""
    blocks = [f"[{result['agent']}] {result['result']}" for result in agent_results]
    return "\n\n".join(blocks)
