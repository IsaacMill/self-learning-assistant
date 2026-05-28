"""FastAPI application for the assistant MVP."""

from __future__ import annotations

from fastapi import FastAPI

from assistant.config import get_settings
from assistant.judge import consistency_judge, factuality_judge, safety_judge, usefulness_judge
from assistant.llm import LLMClient
from assistant.memory import MemoryStore
from assistant.reflection import reflect
from assistant.retrieval import MemoryRetriever
from assistant.schemas import ChatRequest, ChatResponse, MemoryItem, MemoryKind
from assistant.safety import is_blocked_action_request, validate_memory_for_storage

app = FastAPI(title="Self-Learning Assistant MVP")


def build_components() -> tuple[MemoryStore, MemoryRetriever, LLMClient]:
    """Construct runtime dependencies."""

    settings = get_settings()
    store = MemoryStore(settings.database_path)
    retriever = MemoryRetriever(store, settings.chroma_path)
    llm = LLMClient(
        api_key=settings.resolved_api_key,
        model=settings.resolved_model,
        base_url=settings.resolved_base_url,
        provider=settings.llm_provider,
    )
    return store, retriever, llm


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run retrieval, answer generation, judges, and reflection."""

    store, retriever, llm = build_components()
    if is_blocked_action_request(request.message):
        answer = (
            "I cannot execute shell commands, delete files, edit my own source, or perform autonomous tool use "
            "in this MVP. I can discuss a safe plan or explain the boundary."
        )
        judges = {
            "factuality": factuality_judge(answer, []),
            "safety": safety_judge(answer),
            "usefulness": usefulness_judge(request.message, answer),
            "consistency": consistency_judge(answer, []),
        }
        store.log_interaction(
            request.message,
            answer,
            {name: result.model_dump() for name, result in judges.items()},
        )
        return ChatResponse(answer=answer, judges=judges)

    used_memories = retriever.search(request.message, limit=5)
    answer = llm.complete(request.message, used_memories)
    judges = {
        "factuality": factuality_judge(answer, used_memories),
        "safety": safety_judge(answer),
        "usefulness": usefulness_judge(request.message, answer),
        "consistency": consistency_judge(answer, used_memories),
    }
    reflection = reflect(request.message, answer, allow_sensitive_memory=request.allow_sensitive_memory)
    episodic = MemoryItem(
        kind=MemoryKind.episodic,
        content=reflection.summary,
        source="interaction",
        confidence=0.6,
        tags=["conversation"],
    )
    if validate_memory_for_storage(episodic, allow_sensitive=request.allow_sensitive_memory).passed:
        store.add_memory(episodic, allow_sensitive=request.allow_sensitive_memory)
    store.log_interaction(
        request.message,
        answer,
        {name: result.model_dump() for name, result in judges.items()},
    )
    return ChatResponse(
        answer=answer,
        judges=judges,
        used_memories=used_memories if request.debug else [],
        proposed_memories=reflection.proposed_memories,
    )
