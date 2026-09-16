from pydantic import BaseModel, Field
from typing import Literal


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """The answer plus the full decision trail that produced it."""

    answer: str
    intent: str
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    # demo | llm | agent | fallback — which path produced the text.
    mode: str
    # rules | agent — who chose the tools: the application, or the model.
    routing: str = "rules"
    provider: str = "demo"
    model: str = ""


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "error"]
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    environment: str
    provider: str
    model: str
    routing: str
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
