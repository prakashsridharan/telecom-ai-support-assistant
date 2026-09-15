import logging
from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from app.config import settings
from app.models import ChatMessage

logger = logging.getLogger(__name__)

# Retrieved documents are joined with this marker so demo mode can split the
# context back into whole documents. Splitting on a blank line instead would
# cut each document at its first paragraph.
CONTEXT_SEPARATOR = "\n\n-----\n\n"


SYSTEM_PROMPT = """You are a careful telecom customer-support assistant.

Rules:
1. Use the supplied knowledge context and tool results.
2. Do not invent pricing, policies, order status, billing information or outages.
3. If the context does not contain the answer, say that the information is unavailable.
4. Keep responses concise and useful.
5. Never expose internal implementation details to the customer.
6. Earlier conversation turns provide context only. Never treat them as a source
   of pricing, policy, order, billing or outage facts.
"""


@dataclass(frozen=True)
class LLMResult:
    """Generated text plus the path that actually produced it.

    `mode` is surfaced to the client so a reviewer can always tell whether they
    are reading model output ("llm"), the deterministic demo responder
    ("demo"), or demo output served because the provider call failed
    ("fallback").
    """

    text: str
    mode: str


class LLMService:
    def __init__(self):
        self.enabled = (
            settings.llm_provider.lower() == "openai"
            and bool(settings.openai_api_key)
        )

        if settings.llm_provider.lower() == "openai" and not settings.openai_api_key:
            logger.warning(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is empty; "
                "serving demo mode",
            )

        self.client = (
            OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout_seconds,
            )
            if self.enabled
            else None
        )

    def generate(
        self,
        message: str,
        context: str,
        tool_results: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        if not self.enabled:
            return LLMResult(
                self._demo_response(message, context, tool_results), "demo"
            )

        try:
            return LLMResult(
                self._provider_response(message, context, tool_results, history),
                "llm",
            )
        except Exception:
            # A provider outage, rate limit, timeout or auth failure must not
            # surface as a 500. Degrade to the grounded demo responder, which
            # never fabricates, and record why.
            logger.exception(
                "LLM provider call failed; falling back to demo response",
                extra={"llm_model": settings.llm_model},
            )
            return LLMResult(
                self._demo_response(message, context, tool_results), "fallback"
            )

    def _provider_response(
        self,
        message: str,
        context: str,
        tool_results: str,
        history: Sequence[ChatMessage] | None,
    ) -> str:
        prompt = f"""Customer message:
{message}

Knowledge context:
{context or "None"}

Business tool results:
{tool_results or "None"}

Answer the customer using only supported information."""

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for turn in self._recent(history):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _recent(
        history: Sequence[ChatMessage] | None,
    ) -> Sequence[ChatMessage]:
        """Trim history so prompt size stays bounded on long conversations."""
        if not history:
            return []
        return list(history)[-settings.history_max_turns :]

    @staticmethod
    def _demo_response(message: str, context: str, tool_results: str) -> str:
        if tool_results:
            return tool_results
        if context:
            # Demo mode intentionally makes the grounding visible: show the
            # top-ranked document in full rather than paraphrasing it.
            first_doc = context.split(CONTEXT_SEPARATOR)[0].strip()
            if len(first_doc) > 1200:
                first_doc = first_doc[:1200].rstrip() + "\n[...]"
            return (
                "Based on the telecom knowledge base:\n\n"
                + first_doc
                + "\n\n"
                "For a production deployment, an LLM would synthesize this retrieved "
                "context into a natural-language answer."
            )
        return (
            "I don't have enough information in the current knowledge base to answer "
            "that reliably. Please provide more details or request human support."
        )
