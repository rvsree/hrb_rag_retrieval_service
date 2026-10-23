"""Resolves "dynamic" into one of the real SEARCH_STRATEGIES keys, once per
query, before retrieval runs - ported as-is from hrb_chatbot_service's
ai/rag_pipeline/query_retrieval/dynamic_strategy_selector.py (this
session's own Phase 252 there). Heuristic, not an LLM call. Its own
file/class, not folded into retriever.py's dispatch, per the user's own
"logically segregate" instruction - true even though the logic itself is
a handful of simple rules."""

import re

_ACRONYM_OR_CODE_PATTERN = re.compile(r"\b([A-Z]{2,}|\d+[A-Za-z]*)\b")
SHORT_QUERY_WORD_COUNT = 4

_BROAD_QUERY_PHRASES = ("tell me about", "what are my options", "overview of", "everything about", "all about")


class DynamicStrategySelector:
    """One public method, `select(query)` - deliberately a class (not a bare
    function) so this stays a single, obvious extension point if the
    heuristic ever needs per-call state later, without changing every call
    site's shape."""

    def select(self, query: str) -> str:
        words = query.split()
        acronym_or_code_hits = len(_ACRONYM_OR_CODE_PATTERN.findall(query))
        lowered = query.lower()

        if any(phrase in lowered for phrase in _BROAD_QUERY_PHRASES):
            return "mmr"

        if len(words) <= SHORT_QUERY_WORD_COUNT and acronym_or_code_hits > 0:
            return "keyword"

        if acronym_or_code_hits > 0:
            return "hybrid"

        return "similarity"


def resolve_dynamic_strategy(query: str) -> str:
    return DynamicStrategySelector().select(query)
