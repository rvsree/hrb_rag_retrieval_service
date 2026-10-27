# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The retrieval half of the HRB chatbot's RAG pipeline, extracted as its own repo out of `hrb_chatbot_service`'s monolith: decompose -> retrieve -> generate, across 3 answer-generation strategies (genai-rag's fixed pipeline, single-agentic-rag's ReAct tool loop, multi-agentic-rag's LangGraph orchestration), all funneling through one shared guardrail/memory/logging core. Ingestion is a separate, already-built sibling service, `hrb_rag_ingestion_service` - not this repo's concern.

Full design lives in `hrb_chatbot_service`'s own `docs/dev-reference/service-boundary-decision.html`, `microservices-rollout-plan.html`, `rag-retrieval/rag-pipeline-current-state.html`, `rag-guardrails-design.html`, and `mcp/mcp-integration-overview.html` - read those before changing anything here.

Python 3.12, FastAPI.

## Scope decisions - resolved 2026-10-22

- **The 3 strategies are NOT forced into one shared abstraction.** They're genuinely different (fixed pipeline vs. ReAct loop vs. LangGraph graph) - only the guardrail/memory/logging core (`ai/rag_core/guarded_pipeline.py` in the original) is actually shared, same as `hrb_chatbot_service`'s own CLAUDE.md already documents.
- **3 orphan admin routes (admin-docs-sync-to-S3, request-metrics, FAQ submissions) stay out of scope here.** None are retrieval-domain; they're a separate, small `hrb_platform_components` follow-up, not a reason to hold this repo back - flagged, not silently resolved.
- **`hrb_chatbot_service`'s own retrieval code stays dormant, not deleted, once each phase here is verified live** - same reversible pattern already proven for `hrb_rag_ingestion_service`'s Lambda cutover (disable the old path, don't remove it, until the new one's been live long enough to trust).
- **Built across phases, not one shot**: Phase 1 genai-rag (simplest of the 3 - fixed pipeline, no loop/graph), Phase 2 single-agentic-rag, Phase 3 multi-agentic-rag + domain agents, Phase 4 HITL + conversations/feedback admin views.

## Commands

```bash
# install
pip install -r requirements.txt

# run
uvicorn src.rag_retrieval.main:app --reload --port 8400

# test
pytest -v
```

## Architecture

Ports `hrb_chatbot_service`'s `ai/rag_pipeline/**` (genai-rag), `ai/agents/**` (single/multi-agentic-rag), `ai/rag_core/` (the shared guardrail/memory/logging core every strategy funnels through), and `ai/search_pre_processing/**` (10 files - guardrails_input, query_decompose, conversation_memory, cache_layer, context_assembly, context_builder, mode_router, query_normalize, search_filter, safe_terms - already well-factored, ported as-is). MCP calls (leave balance, etc.) go through the same real app-to-app OAuth2 client-credentials flow `hrb-common-libs`' `auth` extra already provides (`OAuth2ClientCredentials`) - `hrb_platform_components`' own Phase 4 already proved this exact pattern once, usable as a direct reference.

Depends on `hrb-common-libs`' `db`, `vectorstore`, `llm`, `mcp`, `auth`, `cache`, and `observability` extras - all already built there, no gaps found during scoping.

## Spec-Driven Development

Same convention as `hrb_chatbot_service`: a spec is a `**Spec:**` block in `ROADMAP.md`'s `## Phases` section, written before that phase's code starts. `.claude/scripts/spec_gate.py` blocks a `Write`/`Edit` under `src/rag_retrieval/**` unless `ROADMAP.md` has an open, spec'd phase.

**Four working norms:**
- **One task at a time.**
- **Flag, don't fix.**
- **Nothing gets pushed/deployed without the user's own review and explicit go-ahead first - every time, not just the first time.**
- **Behavior parity for every extracted phase** - the point of an extraction is that it doesn't change behavior; a ported test must pass the same way it did in `hrb_chatbot_service`, not a looser approximation.

**Available skills**: `session-start`, `delivery-note`, `spec-new`, `spec-implement`.
**Subagents**: `.claude/agents/implementer.md`, `.claude/agents/code-reviewer.md`.
