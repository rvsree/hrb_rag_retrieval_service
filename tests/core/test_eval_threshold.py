from rag_retrieval.core.eval_threshold import build_eval_scores, check_eval_threshold, run_enabled_eval_judges


def test_check_eval_threshold_no_threshold_means_no_check():
    assert check_eval_threshold({"groundedness": 0.1}, None) == (False, [])


def test_check_eval_threshold_flags_breached_metrics():
    breached, metrics = check_eval_threshold({"groundedness": 0.2, "completeness": 0.9}, 0.5)
    assert breached is True
    assert metrics == ["groundedness"]


def test_build_eval_scores_is_none_safe():
    scores = build_eval_scores(None, None, None, None, None)
    assert scores["groundedness"] is None
    assert scores["below_threshold"] is False


async def test_run_enabled_eval_judges_is_deferred_returns_all_none():
    result = await run_enabled_eval_judges("q", "a", ["ctx"], ["ctx"], None, None)
    assert result == (None, None, None, None)
