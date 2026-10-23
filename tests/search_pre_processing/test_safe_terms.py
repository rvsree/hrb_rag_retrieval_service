from rag_retrieval.search_pre_processing.safe_terms import protect_known_safe_terms


def test_wraps_known_safe_term():
    protected = protect_known_safe_terms("Roth 401k contributions")
    assert "Roth" in protected
    assert protected != "Roth 401k contributions"


def test_leaves_unrelated_text_unchanged():
    assert protect_known_safe_terms("plain text with no safe terms") == "plain text with no safe terms"
