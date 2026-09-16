"""The provider boundary.

Everything above this line (orchestrator, routes, tools) is vendor-neutral.
Everything below it (anthropic.py, openai.py) knows exactly one SDK.

Two capabilities, deliberately separate:

- `generate()` — phrase an answer from evidence the caller already gathered.
  This is the rule-based path: the application decided what to retrieve and
  which tool to call, and the model only writes the reply.
- `run_agent()` — the model itself decides which tools to call, in a loop.
  This is real function calling. Providers without tool support raise.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from app.models import ChatMessage
from app.services.tools import ToolBox

# Retrieved documents are joined with this marker so demo mode can split the
# context back into whole documents. Splitting on a blank line instead would
# cut each document at its first paragraph.
CONTEXT_SEPARATOR = "\n\n-----\n\n"


SYSTEM_PROMPT = """You are a telecom customer-support assistant.

Rules:
1. Answer only from the supplied knowledge context and tool results.
2. Never invent pricing, policies, order status, billing details or outages.
3. If the evidence does not answer the question, say the information is
   unavailable and offer to escalate to a human agent.
4. Keep replies short and specific. No preamble.
5. Never mention tools, retrieval, context, or any internal mechanism to the
   customer.
6. Earlier conversation turns are context only, never a source of facts.
"""

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You have tools for looking up orders, billing accounts, service status, and the
knowledge base. Additional rules for using them:

7. Call a lookup tool only with an identifier the customer actually gave you.
   If one is missing, ask the customer for it — never guess, and never reuse an
   example identifier you yourself mentioned earlier.
8. For general "how does this work" questions, search the knowledge base rather
   than guessing from memory.
9. If a tool reports that a record does not exist, tell the customer it was not
   found. Do not substitute a different record or describe a plausible status.
"""


@dataclass(frozen=True)
class LLMResult:
    """A generated reply plus the decision trail that produced it.

    `mode` records which path actually answered:
      "demo"     — the deterministic responder
      "llm"      — the provider phrased a reply from supplied evidence
      "agent"    — the provider chose and called tools itself
      "fallback" — a provider call failed and demo output was served instead
    """

    text: str
    mode: str
    tool_calls: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class ChatProvider(ABC):
    """One model backend."""

    name: str = "unknown"
    model: str = ""
    supports_tool_calling: bool = False

    @abstractmethod
    def generate(
        self,
        message: str,
        context: str,
        tool_results: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        """Phrase an answer from evidence the caller already gathered."""

    def run_agent(
        self,
        message: str,
        history: Sequence[ChatMessage] | None,
        toolbox: ToolBox,
    ) -> LLMResult:
        """Let the model choose and call tools until it can answer."""
        raise NotImplementedError(
            f"The {self.name} provider does not support tool calling."
        )

    @staticmethod
    def _recent(
        history: Sequence[ChatMessage] | None, limit: int
    ) -> list[ChatMessage]:
        """Trim history so prompt size stays bounded on long conversations."""
        if not history:
            return []
        return list(history)[-limit:]

    @staticmethod
    def build_prompt(message: str, context: str, tool_results: str) -> str:
        return f"""Customer message:
{message}

Knowledge context:
{context or "None"}

Business tool results:
{tool_results or "None"}

Answer the customer using only supported information."""


def demo_text(message: str, context: str, tool_results: str) -> str:
    """The deterministic responder.

    Shared by the demo provider and by every provider's failure path, so a
    degraded answer is still grounded rather than absent.
    """
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
