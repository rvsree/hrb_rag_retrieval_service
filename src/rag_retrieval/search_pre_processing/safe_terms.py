"""Known Presidio PERSON false-positives for this domain - ported as-is from
hrb_chatbot_service's ai/search_pre_processing/safe_terms.py."""

_LRI = "⁦"  # nosec B613
_PDI = "⁩"  # nosec B613

SAFE_TERMS_NOT_PII = ["Roth", "Empower"]


def protect_known_safe_terms(text: str) -> str:
    protected = text
    for term in SAFE_TERMS_NOT_PII:
        protected = protected.replace(term, f"{_LRI}{term}{_PDI}")
    return protected
