"""Shared "is this live answer's eval score below threshold" check + the
per-judge on/off switches - ported from hrb_chatbot_service's
ai/rag_core/eval_threshold.py.

DEFERRED, not ported this phase: the real DeepEval live-judge scoring
(evaluations/golden_dataset_harness.py's evaluate_groundedness/
evaluate_completeness/evaluate_retrieval_relevance/evaluate_answer_relevancy,
and deepeval_judge_model.py's GatewayDeepEvalModel) - a real, sizeable
DeepEval integration, judged too large to pull into this phase alongside
everything else (the spec's own "Open questions" flagged this as a
default-defer if it pulls in extra surface area). run_enabled_eval_judges()
below returns (None, None, None, None) unconditionally - eval_scores simply
won't populate (every EvalScores field is Optional/None-safe), a graceful
degradation, not a crash. A follow-up phase should port the real judges."""

from hrb_common.observability.secrets import read_secret

EVAL_THRESHOLD_METRICS = ("groundedness", "completeness", "retrieval_precision")


def check_eval_threshold(eval_scores: dict, threshold: float | None) -> tuple[bool, list[str]]:
    if threshold is None:
        return False, []

    breached = [
        metric
        for metric in EVAL_THRESHOLD_METRICS
        if eval_scores.get(metric) is not None and eval_scores[metric] < threshold
    ]
    return bool(breached), breached


def _is_enabled(env_var_name: str, default: str = "true") -> bool:
    return (read_secret(env_var_name, default) or default).strip().lower() == "true"


def groundedness_enabled() -> bool:
    return _is_enabled("EVAL_GROUNDEDNESS_ENABLED")


def completeness_enabled() -> bool:
    return _is_enabled("EVAL_COMPLETENESS_ENABLED")


def retrieval_relevance_enabled() -> bool:
    return _is_enabled("EVAL_RETRIEVAL_RELEVANCE_ENABLED")


def answer_relevancy_enabled() -> bool:
    return _is_enabled("EVAL_ANSWER_RELEVANCY_ENABLED", default="false")


async def run_enabled_eval_judges(
    query: str,
    answer: str,
    groundedness_context: list[str],
    retrieval_context: list[str],
    judge_model: str | None,
    wider_pool_texts: list[str] | None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """DEFERRED this phase - see module docstring. Always returns
    (None, None, None, None) regardless of the *_enabled() toggles above;
    those toggles are kept (and still read real env config) so a future
    phase wiring in the real judges doesn't also need to redo this plumbing."""
    return None, None, None, None


def build_eval_scores(
    groundedness: dict | None,
    completeness: dict | None,
    retrieval: dict | None,
    eval_score_threshold: float | None,
    answer_relevancy: dict | None = None,
) -> dict:
    eval_scores = {
        "groundedness": groundedness["score"] if groundedness else None,
        "groundedness_verdict": groundedness["verdict"] if groundedness else None,
        "groundedness_explanation": groundedness["explanation"] if groundedness else None,
        "groundedness_prompt": groundedness["prompt"] if groundedness else None,
        "completeness": completeness["score"] if completeness else None,
        "completeness_verdict": completeness["verdict"] if completeness else None,
        "completeness_explanation": completeness["explanation"] if completeness else None,
        "completeness_prompt": completeness["prompt"] if completeness else None,
        "retrieval_precision": retrieval["precision"] if retrieval else None,
        "retrieval_relevant_chunk_indices": retrieval["relevant_chunk_indices"] if retrieval else None,
        "retrieval_precision_explanation": retrieval["explanation"] if retrieval else None,
        "retrieval_precision_prompt": retrieval["prompt"] if retrieval else None,
        "retrieval_recall_estimate": retrieval["recall_estimate"] if retrieval else None,
        "retrieval_f1_estimate": retrieval["f1_estimate"] if retrieval else None,
        "retrieval_recall_pool_size": retrieval["pool_size"] if retrieval else None,
        "retrieval_recall_relevant_count": retrieval["relevant_pool_count"] if retrieval else None,
        "answer_relevancy": answer_relevancy["score"] if answer_relevancy else None,
        "answer_relevancy_verdict": answer_relevancy["verdict"] if answer_relevancy else None,
        "answer_relevancy_explanation": answer_relevancy["explanation"] if answer_relevancy else None,
        "answer_relevancy_prompt": answer_relevancy["prompt"] if answer_relevancy else None,
        "judge_model_used": (groundedness or completeness or retrieval or answer_relevancy or {}).get("model_used"),
    }
    eval_scores["below_threshold"], eval_scores["below_threshold_metrics"] = check_eval_threshold(
        eval_scores, eval_score_threshold
    )
    return eval_scores
