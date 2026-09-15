import logging
import time
import uuid

from fastapi import APIRouter

from app.config import settings
from app.models import (
    ChatRequest,
    ChatResponse,
    ComponentHealth,
    HealthResponse,
)
from app.services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator = Orchestrator()


@router.get("/health", response_model=HealthResponse)
def health():
    """Readiness, not just liveness.

    Reports whether the knowledge base actually loaded and which generation
    path is active, so a deployment with a missing knowledge_base/ directory or
    a silently-ignored API key is visible without opening the chat UI.
    """
    retriever = orchestrator.retriever
    llm = orchestrator.llm

    components = {}

    if retriever.is_ready:
        components["retriever"] = ComponentHealth(
            status="ok",
            detail=f"{retriever.document_count} knowledge documents loaded",
        )
    else:
        components["retriever"] = ComponentHealth(
            status="error",
            detail=f"No knowledge documents found in {retriever.knowledge_dir}",
        )

    if llm.enabled:
        components["llm"] = ComponentHealth(
            status="ok", detail=f"Provider mode using {settings.llm_model}"
        )
    elif settings.llm_provider.lower() == "openai":
        # Configured for a provider but unusable: a real misconfiguration.
        components["llm"] = ComponentHealth(
            status="degraded",
            detail="LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set",
        )
    else:
        components["llm"] = ComponentHealth(
            status="ok", detail="Demo mode; no LLM provider configured"
        )

    statuses = {c.status for c in components.values()}
    overall = (
        "error" if "error" in statuses
        else "degraded" if "degraded" in statuses
        else "ok"
    )

    return HealthResponse(
        status=overall,
        environment=settings.environment,
        mode="llm" if llm.enabled else "demo",
        components=components,
    )


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()

    logger.info(
        "Chat request received",
        extra={
            "request_id": request_id,
            "message_length": len(request.message),
            "history_turns": len(request.history),
        },
    )

    result = orchestrator.respond(request.message, request.history)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    logger.info(
        "Chat request completed",
        extra={
            "request_id": request_id,
            "intent": result["intent"],
            "mode": result["mode"],
            "tool_calls": result["tool_calls"],
            "sources": result["sources"],
            "latency_ms": elapsed_ms,
        },
    )

    return ChatResponse(**result)
