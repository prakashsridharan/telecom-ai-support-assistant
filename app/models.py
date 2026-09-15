from pydantic import BaseModel, Field
from typing import Literal


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    mode: str


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "error"]
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    environment: str
    mode: str
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
