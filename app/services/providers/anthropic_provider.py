"""Claude provider — Anthropic Messages API, with real tool calling.

Agent mode here is genuine function calling: Claude receives the tool schemas
from `app.services.tools` and decides which to call, with what arguments, over
as many turns as it needs. The application executes the calls and feeds results
back. Nothing routes by regex on this path.

A manual loop is used rather than the SDK's beta tool runner for two reasons:
the decision trail (which tools actually ran, which documents they returned)
has to be captured turn by turn for the API response, and the demo is meant to
be readable end to end without a beta dependency.
"""

import logging
from typing import Sequence

import anthropic

from app.config import settings
from app.models import ChatMessage
from app.services.providers.base import (
    AGENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ChatProvider,
    LLMResult,
    demo_text,
)
from app.services.tools import TOOL_SPECS, ToolBox

logger = logging.getLogger(__name__)

REFUSAL_REPLY = (
    "I'm not able to help with that request. If you think this is a mistake, "
    "please ask for a human agent."
)


def _tool_definitions() -> list[dict]:
    """Render the shared specs into Anthropic's tool format.

    `strict` makes the API guarantee arguments validate against the schema, so
    a tool never receives a malformed identifier.
    """
    return [
        {
            "name": spec["name"],
            "description": spec["description"],
            "input_schema": spec["input_schema"],
            "strict": True,
        }
        for spec in TOOL_SPECS
    ]


class AnthropicProvider(ChatProvider):
    name = "anthropic"
    supports_tool_calling = True

    def __init__(self, client=None):
        self.model = settings.anthropic_model
        # Injectable so the tool loop can be tested without a network call.
        self.client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Rule-based path: the caller gathered the evidence, Claude phrases it.
    # ------------------------------------------------------------------

    def generate(
        self,
        message: str,
        context: str,
        tool_results: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        messages: list[dict] = [
            {"role": turn.role, "content": turn.content}
            for turn in self._recent(history, settings.history_max_turns)
        ]
        messages.append(
            {"role": "user", "content": self.build_prompt(message, context, tool_results)}
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=settings.llm_max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": settings.llm_effort},
                messages=messages,
            )
        except Exception:
            logger.exception(
                "Anthropic call failed; serving demo response",
                extra={"model": self.model},
            )
            return LLMResult(demo_text(message, context, tool_results), "fallback")

        if response.stop_reason == "refusal":
            logger.warning(
                "Anthropic refused the request",
                extra={"model": self.model, "stop_details": str(response.stop_details)},
            )
            return LLMResult(REFUSAL_REPLY, "llm")

        return LLMResult(self._text_of(response), "llm")

    # ------------------------------------------------------------------
    # Agent path: Claude chooses the tools.
    # ------------------------------------------------------------------

    def run_agent(
        self,
        message: str,
        history: Sequence[ChatMessage] | None,
        toolbox: ToolBox,
    ) -> LLMResult:
        messages: list[dict] = [
            {"role": turn.role, "content": turn.content}
            for turn in self._recent(history, settings.history_max_turns)
        ]
        messages.append({"role": "user", "content": message})

        tools = _tool_definitions()
        executed: list[str] = []
        sources: list[str] = []

        for iteration in range(settings.agent_max_iterations):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=settings.llm_max_tokens,
                    system=AGENT_SYSTEM_PROMPT,
                    output_config={"effort": settings.llm_effort},
                    tools=tools,
                    messages=messages,
                )
            except Exception:
                logger.exception(
                    "Anthropic agent call failed",
                    extra={"model": self.model, "iteration": iteration},
                )
                return self._degrade(message, toolbox, executed, sources)

            if response.stop_reason == "refusal":
                logger.warning(
                    "Anthropic refused the request",
                    extra={"model": self.model},
                )
                return LLMResult(REFUSAL_REPLY, "agent", executed, sources)

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # No more tool calls: Claude has its answer.
                return LLMResult(
                    self._text_of(response), "agent", executed, sources
                )

            messages.append({"role": "assistant", "content": response.content})

            # All results for one assistant turn go back in a single user
            # message. Splitting them teaches the model to stop calling tools
            # in parallel.
            results = []
            for block in tool_uses:
                # Tool inputs are parsed JSON from the SDK; never string-match
                # on the serialized form.
                arguments = block.input if isinstance(block.input, dict) else {}
                outcome = toolbox.execute(block.name, arguments)
                executed.append(block.name)
                sources.extend(s for s in outcome.sources if s not in sources)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": outcome.text,
                        "is_error": not outcome.found,
                    }
                )
            messages.append({"role": "user", "content": results})

        logger.warning(
            "Agent hit the iteration cap",
            extra={"model": self.model, "limit": settings.agent_max_iterations},
        )
        return self._degrade(message, toolbox, executed, sources)

    # ------------------------------------------------------------------

    def _degrade(
        self,
        message: str,
        toolbox: ToolBox,
        executed: list[str],
        sources: list[str],
    ) -> LLMResult:
        """Answer from the knowledge base when the model path is unusable.

        A provider outage must not become a 500, and must not become an
        unsupported answer either — so fall back to retrieval, which is
        grounded by construction.
        """
        outcome = toolbox.execute("search_knowledge_base", {"query": message})
        if outcome.found:
            merged = sources + [s for s in outcome.sources if s not in sources]
            return LLMResult(demo_text(message, outcome.text, ""), "fallback", executed, merged)
        return LLMResult(demo_text(message, "", ""), "fallback", executed, sources)

    @staticmethod
    def _text_of(response) -> str:
        parts = [b.text for b in response.content if b.type == "text"]
        text = "\n".join(p.strip() for p in parts if p.strip())
        return text or (
            "I don't have enough information to answer that reliably. "
            "Please request human support."
        )
