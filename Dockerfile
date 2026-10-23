# rag_retrieval_service, as a container image.
#
# Build it from the REPOSITORY ROOT, not from anywhere else:
#
#     docker build -t rag-retrieval-service:local .
#
# Two stages, same shape as hrb_api_gateway_service's own Dockerfile.

# ---------------------------------------------------------------------------
# Stage 1 - build the virtual environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# git is needed here only because requirements.txt pins hrb-common-libs via
# a git+https URL, not a published PyPI package.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# Presidio's analyzer (used by the "mask sensitive data" guardrail) needs
# this model at runtime - exactly hrb_chatbot_service's own Phase 82 bug
# ("every query blocked in production" until this line existed). Installs
# as a package into the venv, so the COPY --from=builder /opt/venv /opt/venv
# below picks it up automatically.
RUN /opt/venv/bin/python -m spacy download en_core_web_lg

# ---------------------------------------------------------------------------
# Stage 2 - the image that actually runs
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8400

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 hrbretrieval

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY --chown=hrbretrieval:hrbretrieval src ./src

# guardrails_config/ (NeMo Guardrails' RailsConfig.yml + prompts.yml) needs
# to be loadable at a real filesystem path at runtime, same as src/ itself -
# already included via the `src ./src` copy above since it lives under
# src/rag_retrieval/guardrails_config/, not a separate top-level directory.

USER hrbretrieval

EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8400/ping || exit 1

# --app-dir src, not `src.rag_retrieval.main:app` - this repo's own
# src/rag_retrieval/ modules import each other as bare `from rag_retrieval
# import ...` (confirmed by reading main.py directly).
CMD ["uvicorn", "rag_retrieval.main:app", "--host", "0.0.0.0", "--port", "8400", "--app-dir", "src"]
