"""Generates a grounded answer from retrieved chunks - ported from
hrb_chatbot_service's ai/rag_pipeline/response_generation/response_generator.py.
GatewayChatModel -> this repo's own gateway.build_chat_llm()."""

import time

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from hrb_common.observability.logger import get_logger
from rag_retrieval.core.response_format import TABULAR_FORMAT_INSTRUCTION, should_use_tabular_format
from rag_retrieval.gateway import build_chat_llm
from rag_retrieval.search_pre_processing.context_builder import build_context_from_chunks

logger = get_logger("genai.response_generator")

NO_CONTEXT_ANSWER = "I don't have any information about that in the knowledge base."

SYSTEM_PROMPT_TEMPLATE = """You are the HR benefits assistant for AmazingBank employees. Use ONLY the context below to answer - if the answer isn't in it, say you don't know rather than guessing. Never invent information not present in the context. Always cite the source document by name after each fact.

Example 1 - direct fact lookup:
Context:
[AmazingBank_Empower_401k_Savings_Plan.txt] Automatic enrollment at 5% contribution rate with annual auto-escalation up to 10%. Employer matching contributions of up to 5% (dollar-for-dollar) after one year of service. Immediate eligibility to participate with no waiting period.

Question: When can I start contributing to my 401(k), and does AmazingBank match?
Answer: You're eligible immediately, with no waiting period - you're auto-enrolled at a 5% contribution rate that auto-escalates to 10% over time. AmazingBank matches up to 5% dollar-for-dollar, but only after one year of service [AmazingBank_Empower_401k_Savings_Plan.txt].

Example 2 - eligibility/rule-based answer:
Context:
[AmazingBank_Paid_TimeOff.txt] To qualify for Parental Leave, employees must be actively employed on a U.S. payroll, regularly scheduled to work 20 or more hours per week, and have benefits-eligible status. New hires are eligible once they meet benefits eligibility.

Question: I just started - am I eligible for parental leave right away?
Answer: Only once you meet benefits-eligible status (regularly scheduled 20+ hours/week, not temporary/seasonal/contingent) - there's no separate waiting period beyond that [AmazingBank_Paid_TimeOff.txt].

Example 3 - tiered/multi-value answer:
Context:
[AmazingBank_Guild_Tuition_Assistance.txt] 100% tuition coverage for select undergraduate and master's degree programs. Up to $7,500 annually for other graduate programs and certificates. Up to $5,250 annually for other undergraduate programs.

Question: How much tuition assistance do I get for a certificate program?
Answer: Up to $7,500 per year, unless it's one of the select degree programs covered at 100% - certificates fall under the graduate/certificate tier, not the fully-covered tier [AmazingBank_Guild_Tuition_Assistance.txt].

Example 4 - refusal when the context doesn't cover it:
Context:
[AmazingBank_Healthcare_Benefits.txt] Medical plans: Aetna, Cigna Healthcare, UnitedHealthcare, Centivo. Dental plans: MetLife, Delta Dental. Vision plan: VSP.

Question: Does AmazingBank offer a monthly gym membership stipend?
Answer: I don't have that information in the available documents - the healthcare benefits context covers medical, dental, and vision plans, but says nothing about a gym stipend.

Now answer using only the context below:

Context:
{context}

{tabular_instruction}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ]
)

CONVERSATIONAL_FALLBACK_PROMPT_TEMPLATE = """You are the HR benefits assistant for AmazingBank employees, continuing an existing conversation. No knowledge-base documents matched this specific question, so answer using ONLY the prior conversation below - summarize or refer back to what was already said. Never introduce a new HR policy fact that isn't already in the conversation history. If the prior conversation doesn't help either, say you don't have that information."""

CONVERSATIONAL_FALLBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONVERSATIONAL_FALLBACK_PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


def _to_result(message: AIMessage) -> dict:
    return {
        "answer": message.content,
        "model_used": message.response_metadata.get("model", "unknown"),
        "token_usage": message.response_metadata.get("token_usage"),
    }


def _render_llm_context_turn(label: str, prompt_template: ChatPromptTemplate, inputs: dict) -> dict:
    messages = prompt_template.format_messages(**inputs)

    system_prompt = None
    chat_history = []
    human_message = None
    for index, message in enumerate(messages):
        is_last = index == len(messages) - 1
        if message.type == "system":
            system_prompt = message.content
        elif is_last:
            human_message = message.content
        else:
            chat_history.append({"role": message.type, "content": message.content})

    return {
        "label": label,
        "system_prompt": system_prompt,
        "human_message": human_message,
        "chat_history": chat_history,
        "response": None,
    }


def generate_answer(
    query: str,
    chunks: list[dict],
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    chat_history: list[BaseMessage] | None = None,
    request_id: str | None = None,
) -> dict:
    """Returns {"answer": str, "model_used": str}."""
    config = {"metadata": {"request_id": request_id}} if request_id else None
    if not chunks:
        if chat_history:
            logger.info(
                "No chunks retrieved for %r but chat_history has %d turn(s) - answering from conversation memory only",
                query, len(chat_history),
            )
            llm = build_chat_llm(model_name=model_name, temperature=temperature)
            chain = CONVERSATIONAL_FALLBACK_PROMPT | llm | RunnableLambda(_to_result)
            fallback_inputs = {"question": query, "chat_history": chat_history}
            started_at = time.perf_counter()
            result = chain.invoke(fallback_inputs, config=config)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
            logger.info("Generated a %d-character answer from conversation history only (0 chunks)", len(result["answer"]))
            llm_context_turn = _render_llm_context_turn("answer", CONVERSATIONAL_FALLBACK_PROMPT, fallback_inputs)
            llm_context_turn["latency_ms"] = latency_ms
            llm_context_turn["response"] = result["answer"]
            result["llm_context_turn"] = llm_context_turn
            return result

        logger.info("No chunks retrieved for %r - returning the no-context answer, not calling the LLM", query)
        return {
            "answer": NO_CONTEXT_ANSWER,
            "model_used": model_name or "none",
            "token_usage": None,
            "llm_context_turn": None,
        }

    llm = build_chat_llm(model_name=model_name, temperature=temperature)
    chain = RAG_PROMPT | llm | RunnableLambda(_to_result)

    context = build_context_from_chunks(chunks)
    tabular_instruction = TABULAR_FORMAT_INSTRUCTION if should_use_tabular_format(query) else ""
    rag_inputs = {"context": context, "question": query, "chat_history": chat_history or [], "tabular_instruction": tabular_instruction}
    started_at = time.perf_counter()
    result = chain.invoke(rag_inputs, config=config)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

    logger.info("Generated a %d-character answer from %d chunk(s)", len(result["answer"]), len(chunks))
    llm_context_turn = _render_llm_context_turn("answer", RAG_PROMPT, rag_inputs)
    llm_context_turn["latency_ms"] = latency_ms
    llm_context_turn["response"] = result["answer"]
    result["llm_context_turn"] = llm_context_turn
    return result
