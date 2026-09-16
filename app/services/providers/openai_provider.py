"""OpenAI provider — Chat Completions, with real tool calling.

Kept alongside the Claude provider on purpose: the point of the provider
boundary is that it holds more than one implementation. Both render the same
`TOOL_SPECS` and return the same `LLMResult`, so the orchestrator, the API
contract and the evaluation harness are identical whichever is configured.
"""

import json
import logging
from typing import Sequence

from openai import OpenAI

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


def _tool_definitions() -> list[dict]:
    """Render the shared specs into OpenAI's function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
                "strict": True,
            },
        }
        for spec in TOOL_SPECS
    ]


class OpenAIProvider(ChatProvider):
    name = "openai"
    supports_tool_calling = True

    def __init__(self, client=None):
        self.model = settings.openai_model
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    def generate(
        self,
        message: str,
        context: str,
        tool_results: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {"role": turn.role, "content": turn.content}
            for turn in self._recent(history, settings.history_max_turns)
        )
        messages.append(
            {"role": "user", "content": self.build_prompt(message, context, tool_results)}
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=settings.llm_max_tokens,
                messages=messages,
            )
        except Exception:
            logger.exception(
                "OpenAI call failed; serving demo response",
                extra={"model": self.model},
            )
            return LLMResult(demo_text(message, context, tool_results), "fallback")

        return LLMResult((response.choices[0].message.content or "").strip(), "llm")

    def run_agent(
        self,
        message: str,
        history: Sequence[ChatMessage] | None,
        toolbox: ToolBox,
    ) -> LLMResult:
        messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        messages.extend(
            {"role": turn.role, "content": turn.content}
            for turn in self._recent(history, settings.history_max_turns)
        )
        messages.append({"role": "user", "content": message})

        tools = _tool_definitions()
        executed: list[str] = []
        sources: list[str] = []

        for iteration in range(settings.agent_max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=settings.llm_max_tokens,
                    tools=tools,
                    messages=messages,
                )
            except Exception:
                logger.exception(
                    "OpenAI agent call failed",
                    extra={"model": self.model, "iteration": iteration},
                )
                return self._degrade(message, toolbox, executed, sources)

            choice = response.choices[0].message
            calls = choice.tool_calls or []
            if not calls:
                return LLMResult(
                    (choice.content or "").strip(), "agent", executed, sources
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )

            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    logger.warning(
                        "Tool arguments were not valid JSON",
                        extra={"tool": call.function.name},
                    )
                    arguments = {}
                outcome = toolbox.execute(call.function.name, arguments)
                executed.append(call.function.name)
                sources.extend(s for s in outcome.sources if s not in sources)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": outcome.text,
                    }
                )

        logger.warning(
            "Agent hit the iteration cap",
            extra={"model": self.model, "limit": settings.agent_max_iterations},
        )
        return self._degrade(message, toolbox, executed, sources)

    def _degrade(
        self,
        message: str,
        toolbox: ToolBox,
        executed: list[str],
        sources: list[str],
    ) -> LLMResult:
        outcome = toolbox.execute("search_knowledge_base", {"query": message})
        if outcome.found:
            merged = sources + [s for s in outcome.sources if s not in sources]
            return LLMResult(
                demo_text(message, outcome.text, ""), "fallback", executed, merged
            )
        return LLMResult(demo_text(message, "", ""), "fallback", executed, sources)
