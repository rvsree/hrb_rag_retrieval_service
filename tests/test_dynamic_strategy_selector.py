from rag_retrieval.genai.dynamic_strategy_selector import resolve_dynamic_strategy


def test_broad_query_phrase_picks_mmr():
    assert resolve_dynamic_strategy("tell me about health benefits") == "mmr"


def test_short_query_with_acronym_picks_keyword():
    assert resolve_dynamic_strategy("401K rules") == "keyword"


def test_long_query_with_acronym_picks_hybrid():
    assert resolve_dynamic_strategy("can you walk me through how the 401K matching works in detail") == "hybrid"


def test_plain_query_picks_similarity():
    assert resolve_dynamic_strategy("how do I enroll in dental coverage") == "similarity"
