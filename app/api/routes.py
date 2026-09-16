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
from app.services.providers import provider_status

logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator = Orchestrator()


@router.get("/health", response_model=HealthResponse)
def health():
    """Readiness, not just liveness.

    Reports whether the knowledge base loaded, which model backend is active,
    and which routing strategy will actually run — so a deployment with a
    missing knowledge_base/ directory, a silently-ignored API key, or agent
    mode quietly degraded to rules is visible without opening the chat UI.
    """
    retriever = orchestrator.retriever
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

    llm_status, llm_detail = provider_status()
    components["llm"] = ComponentHealth(status=llm_status, detail=llm_detail)

    requested_routing = settings.routing_mode.lower()
    effective_routing = orchestrator.routing_mode
    if requested_routing == effective_routing:
        components["routing"] = ComponentHealth(
            status="ok", detail=f"Routing mode '{effective_routing}'"
        )
    else:
        components["routing"] = ComponentHealth(
            status="degraded",
            detail=(
                f"Routing mode '{requested_routing}' requested but the active "
                f"provider cannot call tools; using '{effective_routing}'"
            ),
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
        provider=orchestrator.llm.name,
        model=orchestrator.llm.model,
        routing=effective_routing,
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
            "routing": result["routing"],
            "provider": result["provider"],
            "model": result["model"],
            "tool_calls": result["tool_calls"],
            "sources": result["sources"],
            "latency_ms": elapsed_ms,
        },
    )

    return ChatResponse(**result)
