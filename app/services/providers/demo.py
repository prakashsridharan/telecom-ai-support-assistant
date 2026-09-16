"""The keyless provider.

Not a stub: it returns real, grounded, useful text and says so. It is what
makes the project runnable by a reviewer with no API key, which is the single
most valuable property of a portfolio piece.

It cannot do agent mode — choosing tools is exactly the thing that needs a
model — so `run_agent` stays unimplemented and the orchestrator falls back to
rule-based routing.
"""

from typing import Sequence

from app.models import ChatMessage
from app.services.providers.base import ChatProvider, LLMResult, demo_text


class DemoProvider(ChatProvider):
    name = "demo"
    model = "deterministic"
    supports_tool_calling = False

    def generate(
        self,
        message: str,
        context: str,
        tool_results: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        return LLMResult(demo_text(message, context, tool_results), "demo")
