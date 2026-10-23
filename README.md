# rag_retrieval_service

Retrieval half of the HRB chatbot RAG pipeline: decompose -> retrieve -> generate, across 3 answer-generation strategies (genai-rag, single-agentic-rag, multi-agentic-rag) sharing one guardrail/memory/logging core. Extracted out of `hrb_chatbot_service`'s monolith.

Full design doc: `hrb_chatbot_service`'s `docs/dev-reference/service-boundary-decision.html` and `docs/dev-reference/microservices-rollout-plan.html`.

**Status, 2026-10-23: built, deployed, independently verified - deliberately NOT yet cut over.** Live on AWS App Runner (`https://mpm3qjkbkc.us-east-1.awsapprunner.com`), sized to match `hrb_chatbot_service`'s own instance (same agentic/LLM workload). `hrb_chatbot_service`'s own retrieval code is still what actually serves production traffic - this service runs standalone, verified independently via a real multi-turn Playwright suite driving the live UI (`hrb_chatbot_ui/e2e/golden-dataset-conversational.spec.ts`, 7/9 passing).

3 real bugs were found live during that testing and fixed, same session (see `ROADMAP.md`'s Phase 5 entry): missing `presidio`/`spacy` guardrail dependencies (every query 422'd), `vector_db`/`search_strategy`/`llm_provider` resolved via a bare Python `or` instead of real env-config resolution (silently ignored production's `pinecone` config), and an `EvalScores` schema mismatch against this repo's own no-op eval judges. 2 real findings remain, not bugs - a conversational follow-up using a pronoun ("is *that* available right away?") doesn't always retrieve the right context, since this repo's retrieval step doesn't rewrite a follow-up against conversation history before the vector search - a known RAG architecture gap, flagged for a deliberate future phase, not cutover-blocking by itself.
