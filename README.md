# rag_retrieval_service

Retrieval half of the HRB chatbot RAG pipeline: decompose -> retrieve -> generate, across 3 answer-generation strategies (genai-rag, single-agentic-rag, multi-agentic-rag) sharing one guardrail/memory/logging core. Extracted out of `hrb_chatbot_service`'s monolith.

Full design doc: `hrb_chatbot_service`'s `docs/dev-reference/service-boundary-decision.html` and `docs/dev-reference/microservices-rollout-plan.html`.

Status: scaffolding only - no implementation yet. See `ROADMAP.md`.
