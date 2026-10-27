# hrb_rag_retrieval_service

Retrieval half of the HRB chatbot RAG pipeline: decompose -> retrieve -> generate, across 3 answer-generation strategies (genai-rag, single-agentic-rag, multi-agentic-rag) sharing one guardrail/memory/logging core. Extracted out of `hrb_chatbot_service`'s monolith. Deployed and independently verified - deliberately not yet in the live request path.

## For reviewers / new team members (5 minutes)

1. Run it locally (below) and hit `GET /ping`.
2. [Current state architecture](../docs/architecture/diagrams/rag-retrieval-service-current-state-architecture.drawio) - decompose/retrieve/guardrails/generate, and what it shares with `hrb_chatbot_service`.
3. [Target state (the cutover itself)](../docs/architecture/diagrams/rag-retrieval-service-target-state-scaling-architecture.drawio) - today vs. target, side by side. This is the single biggest open architecture decision in the whole platform right now.
4. [Deployment architecture](../docs/architecture/diagrams/rag-retrieval-service-deployment-architecture.drawio) - the actual AWS resources.

## Live demo

Deployed and independently verified, but **not** what's actually answering
questions at **[hrb-chatbot-ui.rvsree.dev](https://hrb-chatbot-ui.rvsree.dev)**
today - `hrb_chatbot_service`'s own retrieval code still serves that traffic
until this repo's cutover (see Status below) is approved. You can hit this
service's own deployed URL directly to see it work standalone.

## Quick start

```bash
pip install -r requirements.txt
uvicorn rag_retrieval.main:app --host 0.0.0.0 --port 8400 --app-dir src --reload
curl http://127.0.0.1:8400/ping
```

Tests: `pytest tests/ -v` from the repo root.

## Tech stack

| Layer | What's actually used |
|---|---|
| Framework | FastAPI + Uvicorn |
| Orchestration | LangChain 0.3.x + LangGraph 0.3.x, with a Postgres checkpointer (`langgraph-checkpoint-postgres`) for durable agent state |
| Retrieval | `langchain-chroma`, `langchain-pinecone` - dense + hybrid BM25, Dynamic strategy routing |
| Guardrails | Presidio + spaCy (`en_core_web_lg`) for PII masking - ported from `hrb_chatbot_service`, same model dependency |
| Shared code | `hrb_common_libs[observability,db,vectorstore,llm,mcp,auth,cache,web,storage,email]` - the richest dependency set of any service in this split, pinned to `v0.4.0` |
| Deployment | AWS App Runner (Docker, 2-stage build), 1024 CPU / 2048 Memory - same size as `hrb_chatbot_service`'s own LLM workload |

## Status, 2026-10-24: built, deployed, independently verified - deliberately NOT yet cut over

Live on AWS App Runner (`https://mpm3qjkbkc.us-east-1.awsapprunner.com`), sized to match `hrb_chatbot_service`'s own instance (same agentic/LLM workload, confirmed via `aws apprunner describe-service`). `hrb_chatbot_service`'s own retrieval code is still what actually serves production traffic - the Gateway's own `BACKEND_SERVICE_URL` secret points there, not here, and `hrb_chatbot_ui`'s single `VITE_API_BASE_URL` means its own e2e suite exercises that path, not this one. This service's own verification is direct: live `POST /query` smoke tests against its own deployed URL, 79/79 unit tests passing. No cutover criteria are defined yet - that's still an open decision, not just a mechanical flip; see the target-state diagram above.

3 bugs were found live during that testing and fixed, same session (see `ROADMAP.md`'s Phase 5 entry): missing `presidio`/`spacy` guardrail dependencies (every query 422'd), `vector_db`/`search_strategy`/`llm_provider` resolved via a bare Python `or` instead of proper env-config resolution (silently ignored production's `pinecone` config), and an `EvalScores` schema mismatch against this repo's own no-op eval judges.

**2 findings remain, not bugs:** a conversational follow-up using a pronoun ("is *that* available right away?") doesn't always retrieve the right context, since this repo's retrieval step doesn't rewrite a follow-up against conversation history before the vector search - a known RAG architecture gap, flagged for a deliberate future phase, not cutover-blocking by itself. And the cutover decision itself (flip `BACKEND_SERVICE_URL`) is still open - see the target-state diagram above for what it unlocks and what it costs `hrb_chatbot_service`.

## Documentation index

| Doc | What it covers |
|---|---|
| [Overview / Spec / Architecture / Design / Decisions / Interview FAQ](../docs/rag-pipeline/retrieval/genai-rag-chat/) | The full HTML doc set for the genai-rag mode (also see `single-agentic-rag/`, `multi-agentic-rag/`, `retrieval-strategies/`, `search-pre-processing-layer/`, `chunking-strategy/` as siblings) |
| [Build process](../docs/deployment-and-ops/build-and-deploy/05a-rag-retrieval-service-build-process.html) | Docker build, step-by-step |
| [Deployment architecture (HTML)](../docs/deployment-and-ops/build-and-deploy/05b-rag-retrieval-service-deployment-architecture.html) | Same content as the `.drawio` above, narrative form |
| [`service-boundary-decision.html`](../docs/architecture/service-boundary-decision.html) | Why this service exists as its own repo |
| [`platform-overview.html`](../docs/architecture/platform-overview.html) | The cross-service hub - business capabilities, all 7 repos, one diagram |

Older paths under `docs/dev-reference/` (no `hrb_chatbot_v3/` prefix) referenced in earlier commits are stale - that folder was reorganized 2026-10-23, and the links above reflect the current location.
