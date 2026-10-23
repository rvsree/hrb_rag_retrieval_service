"""FastAPI entrypoint for this repo - genai-rag's /query, single-agentic-rag's
/single-agentic-rag/query, multi-agentic-rag's /multi-agentic-rag/query,
(Phase 4) HITL's /unpaid-leave/*, and the 3 retrieval-domain admin views
under /admin/*, mounted at the repo root (no /v1/... prefix wiring yet -
this repo has no Gateway in front of it)."""

from fastapi import FastAPI

from rag_retrieval.admin.routes import router as admin_router
from rag_retrieval.hitl.routes import router as hitl_router
from rag_retrieval.multi_agentic_routes import router as multi_agentic_router
from rag_retrieval.routes import router
from rag_retrieval.single_agentic_routes import router as single_agentic_router

app = FastAPI(title="rag_retrieval_service")
app.include_router(router)
app.include_router(single_agentic_router)
app.include_router(multi_agentic_router)
app.include_router(hitl_router)
app.include_router(admin_router)


@app.get("/ping")
async def ping():
    return {"status": "ok"}
