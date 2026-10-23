"""Request/response contracts for the genai-rag query API - ported from
hrb_chatbot_service's models/rag.py + models/common.py + common/enums.py's
SearchStrategy/VectorDB/LlmProvider, same field shapes/validation."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Ported from hrb_chatbot_service's common/enums.py - only the values
    access_control.py's ACL bypass logic needs (HR_SUPPORT/ADMIN always see
    every document). RBAC enforcement itself (require_role) isn't ported
    this phase - flagged in the delivery note."""

    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR_SUPPORT = "hr_support"
    ADMIN = "admin"
    EMPLOYEE_ADMIN = "employee_admin"


class VectorDB(StrEnum):
    CHROMADB = "chromadb"
    PINECONE = "pinecone"


class LlmProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"


class SearchStrategy(StrEnum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    KNN = "knn"
    DYNAMIC = "dynamic"


class UserProfile(BaseModel):
    employee_id: str = Field(..., max_length=50)
    full_name: str = Field(..., max_length=200)
    role: str = Field(..., max_length=50)
    department: str | None = Field(None, max_length=100)


class ToolCallInfo(BaseModel):
    tool_name: str
    tool_input: str
    tool_type: str = "other"
    latency_ms: float | None = None
    success: bool = True
    iteration: int | None = None
    search_strategy: str | None = None


class SearchOptions(BaseModel):
    top_k: int | None = Field(None, ge=1, le=20)
    vector_db: VectorDB | None = None
    search_strategy: SearchStrategy | None = None
    lambda_mult: float | None = Field(None, ge=0.0, le=1.0)
    use_multi_query: bool = False
    use_self_query: bool = False
    llm_provider: LlmProvider | None = None
    enable_corrective_retry: bool = False


class GenerationOptions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str | None = Field(None, max_length=100)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1)


class EvalOptions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    judge_model: str | None = Field(None, max_length=100)
    eval_score_threshold: float | None = Field(None, ge=0.0, le=1.0)


class ClassificationOptions(BaseModel):
    """decompose_model: genai-rag only. planner_model: multi-agentic-rag
    only (Phase 3) - each pipeline has its own single classification step,
    not a shared one."""

    model_config = ConfigDict(protected_namespaces=())

    decompose_model: str | None = Field(None, max_length=100)
    planner_model: str | None = Field(None, max_length=100)


class RagQueryRequest(BaseModel):
    user_profile: UserProfile | None = None
    query: str = Field(..., min_length=1, max_length=2000)
    search_options: SearchOptions | None = None
    generation_options: GenerationOptions | None = None
    eval_options: EvalOptions | None = None
    classification_options: ClassificationOptions | None = None
    guardrails_model: str | None = Field(None, max_length=100)
    enable_conversation_memory: bool = False
    conversation_id: str | None = None


class RetrievedChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float | None = None
    matched_by: list[str] | None = None


class AnswerInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    model_used: str


class RetrievalInfo(BaseModel):
    vector_db: str
    search_strategy: str
    applied_filter: dict | None = None
    acl_denied_count: int = 0
    sources: list[RetrievedChunk]


class LatencyInfo(BaseModel):
    total: float
    retrieval: float | None = None
    generation: float | None = None
    eval: float | None = None
    decomposition: float | None = None


class TokenUsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EvalScores(BaseModel):
    groundedness: float | None = None
    groundedness_verdict: str | None = None
    groundedness_explanation: str | None = None
    groundedness_prompt: str | None = None
    completeness: float | None = None
    completeness_verdict: str | None = None
    completeness_explanation: str | None = None
    completeness_prompt: str | None = None
    retrieval_precision: float | None = None
    retrieval_relevant_chunk_indices: list[int] = Field(default_factory=list)
    retrieval_precision_explanation: str | None = None
    retrieval_precision_prompt: str | None = None
    retrieval_recall_estimate: float | None = None
    retrieval_f1_estimate: float | None = None
    retrieval_recall_pool_size: int | None = None
    retrieval_recall_relevant_count: int | None = None
    retrieval_retry_attempted: bool = False
    answer_relevancy: float | None = None
    answer_relevancy_verdict: str | None = None
    answer_relevancy_explanation: str | None = None
    answer_relevancy_prompt: str | None = None
    judge_model_used: str | None = None
    below_threshold: bool = False
    below_threshold_metrics: list[str] = Field(default_factory=list)


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class McpToolCallDetail(BaseModel):
    tool_name: str
    arguments: dict
    raw_result: list[str]


class LlmContextTurn(BaseModel):
    label: str
    latency_ms: float | None = None
    system_prompt: str | None = None
    human_message: str | None = None
    chat_history: list[ChatHistoryMessage] = Field(default_factory=list)
    response: str | None = None


class LlmContextInfo(BaseModel):
    turns: list[LlmContextTurn] = Field(default_factory=list)
    mcp_tool_calls: list[McpToolCallDetail] = Field(default_factory=list)


class ActivatedRailInfo(BaseModel):
    name: str
    decisions: list[str] = Field(default_factory=list)
    stop: bool = False


class GuardrailCheckStatus(BaseModel):
    status: str
    rail: str | None = None
    activated_rails: list[ActivatedRailInfo] = Field(default_factory=list)


class GuardrailInfo(BaseModel):
    input: GuardrailCheckStatus
    output: GuardrailCheckStatus


class ExplainabilityInfo(BaseModel):
    request_id: str
    served_from_cache: bool
    llm_call_count: int
    latency_ms: LatencyInfo
    token_usage: TokenUsageInfo | None = None
    llm_context: LlmContextInfo | None = None
    routed_to: str | None = None
    eval_scores: EvalScores | None = None
    temperature: float | None = None
    guardrails: GuardrailInfo
    decompose_model_used: str | None = None
    planner_model_used: str | None = None
    guardrails_model_used: str | None = None


class RagQueryResponse(BaseModel):
    query: str
    answer_info: AnswerInfo
    retrieval_info: RetrievalInfo
    explainability_info: ExplainabilityInfo
    tools_used: list[ToolCallInfo] = Field(default_factory=list)
    conversation_id: str | None = None


class AgenticRagRequest(BaseModel):
    """Phase 2 - single-agentic-rag's own request shape, ported from
    hrb_chatbot_service's models/agentic_rag.py. search_strategy is
    narrower than genai-rag's own SearchOptions.search_strategy on
    purpose - single_agentic/tools.py's search_knowledge_base() only ever
    understands "similarity"/"mmr"."""

    user_profile: UserProfile | None = None
    query: str = Field(..., min_length=1, max_length=2000)
    max_iterations: int | None = Field(None, ge=1, le=10)
    search_strategy: SearchStrategy | None = None
    eval_options: EvalOptions | None = None
    guardrails_model: str | None = Field(None, max_length=100)
    enable_conversation_memory: bool = False
    conversation_id: str | None = None


class AgenticRagResponse(BaseModel):
    query: str
    answer: str
    tools_used: list[ToolCallInfo] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    iterations: int
    explainability_info: ExplainabilityInfo
    conversation_id: str | None = None


class MultiAgenticRagRequest(BaseModel):
    """Phase 3 - multi-agentic-rag's own request shape, ported from
    hrb_chatbot_service's models/multi_agentic_rag.py. search_strategy is
    narrower than genai-rag's own SearchOptions.search_strategy on purpose -
    multi_agentic/domain_agents/vector_kb_agent.py only ever understands
    "similarity"/"mmr"."""

    user_profile: UserProfile | None = None
    query: str = Field(..., min_length=1, max_length=2000)
    search_strategy: SearchStrategy | None = None
    eval_options: EvalOptions | None = None
    classification_options: ClassificationOptions | None = None
    guardrails_model: str | None = Field(None, max_length=100)
    enable_conversation_memory: bool = False
    conversation_id: str | None = None


class AgentTaskInfo(BaseModel):
    agent: str
    focus: str
    reasoning: str


class MultiAgenticRagResponse(BaseModel):
    query: str
    answer: str
    tasks: list[AgentTaskInfo]
    suggested_follow_up_questions: list[str] = Field(default_factory=list)
    tools_used: list[ToolCallInfo] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    iterations: int
    explainability_info: ExplainabilityInfo
    conversation_id: str | None = None


# --- Phase 4: HITL (unpaid leave) + admin views - ported from
# hrb_chatbot_service's models/unpaid_leave.py + models/admin.py (the 3
# retrieval-domain route groups only) + models/conversations.py +
# models/feedback.py. No vector_db/search_strategy fields - this is a
# structured form submission, not a RAG question.


class LeaveTypeCode(StrEnum):
    UNPAID = "UNPAID"
    CRITICAL_CAREGIVER = "CRITICAL_CAREGIVER"
    STD = "STD"
    LTD = "LTD"


class LeaveDocumentUploadUrlRequest(BaseModel):
    user_profile: UserProfile
    filename: str
    content_type: str


class LeaveDocumentUploadUrlResponse(BaseModel):
    upload_url: str
    key: str
    expires_in_seconds: int


class UnpaidLeaveRequestBody(BaseModel):
    user_profile: UserProfile
    start_date: str
    end_date: str
    purpose: str = Field(..., min_length=1, max_length=2000)
    leave_type_code: LeaveTypeCode = LeaveTypeCode.UNPAID
    document_reference: str | None = None


class UnpaidLeaveRequestResponse(BaseModel):
    request_id: int
    employee_id: str
    leave_type: str
    status: str
    is_repeat_within_12_months: bool
    hitl_request_context_id: int
    hitl_status: str
    days: float


class HitlRequestSummary(BaseModel):
    hitl_request_context_id: int
    request_type: str
    employee_id: str | None
    query: str
    context: dict
    status: str
    assigned_to: str | None
    created_at: str | None


class PendingHitlRequestsResponse(BaseModel):
    total: int
    requests: list[HitlRequestSummary]


class ResolveHitlRequestBody(BaseModel):
    user_profile: UserProfile
    hitl_request_context_id: int
    decision: str
    response_text: str = Field(..., min_length=1, max_length=2000)


class ResolveHitlRequestResponse(BaseModel):
    hitl_request_context_id: int
    status: str
    decision: str
    response: str
    related_leave_request_updated: bool


class LeaveDocumentDownloadUrlResponse(BaseModel):
    download_url: str
    original_filename: str
    expires_in_seconds: int


class MyLeaveRequestSummary(BaseModel):
    request_id: int
    leave_type: str
    start_date: str
    end_date: str
    days: float
    purpose: str
    status: str
    decision: str | None
    hr_comments: str | None
    hr_specialist_name: str | None
    created_at: str | None
    responded_at: str | None
    task_state: str | None = None
    review_brief: str | None = None
    decision_message: dict | None = None
    produced_artifacts: list[dict] = Field(default_factory=list)


class MyLeaveRequestsResponse(BaseModel):
    total: int
    requests: list[MyLeaveRequestSummary]


class HrLeaveRequestSummary(BaseModel):
    hitl_request_context_id: int
    request_id: int
    employee_id: str | None
    employee_name: str | None
    leave_type: str
    start_date: str
    end_date: str
    days: float
    purpose: str
    status: str
    decision: str | None
    hr_comments: str | None
    hr_specialist_name: str | None
    created_at: str | None
    responded_at: str | None
    task_state: str | None = None
    review_brief: str | None = None
    decision_message: dict | None = None
    produced_artifacts: list[dict] = Field(default_factory=list)


class HrLeaveRequestsResponse(BaseModel):
    total: int
    requests: list[HrLeaveRequestSummary]


class AdminConversationSummary(BaseModel):
    conversation_id: str
    employee_id: str | None
    title: str
    started_at: str
    last_updated_at: str


class AdminConversationListResponse(BaseModel):
    count: int
    conversations: list[AdminConversationSummary]


class AdminDeleteConversationResponse(BaseModel):
    conversation_id: str
    turns_deleted: int


class ConversationTurn(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    turns: list[ConversationTurn]


class FeedbackRecord(BaseModel):
    id: int
    employee_id: str
    conversation_id: str | None
    message_id: str
    vote: str
    reason_tags: list[str]
    notes: str | None
    question: str
    answer: str
    created_at: str


class FeedbackListResponse(BaseModel):
    count: int
    feedback: list[FeedbackRecord]


class AdminDeleteFeedbackResponse(BaseModel):
    feedback_id: int
    deleted: bool


class AdminDeleteLeaveRequestResponse(BaseModel):
    hitl_request_context_id: int
    deleted_hitl_request: bool
    deleted_specialist_leave_reviews: int
    deleted_leave_request: bool
    deleted_document_count: int
