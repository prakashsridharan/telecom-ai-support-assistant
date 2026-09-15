import logging
import re
from typing import Sequence

from app.config import settings
from app.models import ChatMessage
from app.services.llm import CONTEXT_SEPARATOR, LLMResult, LLMService
from app.services.retriever import KnowledgeRetriever
from app.tools.telecom_tools import (
    get_billing_status,
    get_order_status,
    get_service_status,
    known_locations,
)

logger = logging.getLogger(__name__)

# Identifiers tolerate "ORD-10234", "ord 10234" and "ord10234".
_ORDER_RE = re.compile(r"\b(ord)[-\s]?(\d{3,})\b", re.IGNORECASE)
_CUSTOMER_RE = re.compile(r"\b(cust)[-\s]?(\d{3,})\b", re.IGNORECASE)

_ORDER_PHRASES = ("order status", "my order", "track my", "where is my order")
_BILLING_PHRASES = ("billing", "bill", "balance", "invoice", "refund", "payment")

_SERVICE_PHRASES = re.compile(
    r"\b(outage|outages|no service|service status|blackout|network status)\b",
    re.IGNORECASE,
)
# "down" on its own over-triggers ("my speed went down", "the price came down"),
# so a fault word only counts as an outage signal when it sits near a network
# noun, or when "down" is followed by a place. Recall lost to this narrowing is
# recovered by the knowledge-base fallback.
_NETWORK_NOUN = (
    r"network|service|signal|coverage|connection|connectivity|broadband|"
    r"internet|line|tower|mobile data"
)
_FAULT_WORD = (
    r"down|outage|issue|issues|problem|problems|not working|interrupted|dropping"
)
_SERVICE_DOWN = re.compile(
    rf"\b({_NETWORK_NOUN})\b[^.?!]{{0,24}}\b({_FAULT_WORD})\b"
    rf"|\b({_FAULT_WORD})\b[^.?!]{{0,24}}\b({_NETWORK_NOUN})\b"
    r"|\bdown\s+(in|at|near|around)\b",
    re.IGNORECASE,
)

# A short message leaning on a referring word is treated as a follow-up and
# inherits the previous turn's intent.
_REFERENCE_WORDS = re.compile(
    r"\b(it|its|it's|that|this|those|these|they|them|there|mine|same|one)\b",
    re.IGNORECASE,
)
_FOLLOWUP_OPENERS = ("what about", "how about", "and what", "and how", "any update")
_FOLLOWUP_MAX_WORDS = 12

# A message naming a knowledge topic stands on its own and must not inherit the
# previous turn's intent: "And what plans do you offer?" opens like a follow-up
# but is a fresh knowledge question.
_KNOWLEDGE_TOPICS = re.compile(
    r"\b(plan|plans|pricing|price|prices|cost|costs|tariff|package|packages|"
    r"roaming|policy|policies|offer|offers|upgrade|downgrade|allowance)\b",
    re.IGNORECASE,
)


class Orchestrator:
    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.llm = LLMService()

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recent(
        history: Sequence[ChatMessage] | None,
    ) -> list[ChatMessage]:
        if not history:
            return []
        return list(history)[-settings.history_max_turns :]

    @classmethod
    def _is_followup(cls, message: str) -> bool:
        text = message.strip().lower()
        if len(text.split()) > _FOLLOWUP_MAX_WORDS:
            return False
        if _KNOWLEDGE_TOPICS.search(text):
            return False
        if text.startswith(_FOLLOWUP_OPENERS):
            return True
        return bool(_REFERENCE_WORDS.search(text))

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_single(message: str) -> str:
        text = message.lower()
        if _ORDER_RE.search(text) or any(p in text for p in _ORDER_PHRASES):
            return "order_status"
        if any(p in text for p in _BILLING_PHRASES):
            return "billing"
        if _SERVICE_PHRASES.search(text) or _SERVICE_DOWN.search(text):
            return "service_status"
        return "knowledge"

    @classmethod
    def classify(
        cls,
        message: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> str:
        """Classify the current message, inheriting intent on short follow-ups.

        "Is it delivered?" carries no intent of its own; when the previous user
        turn was an order lookup, the follow-up is one too.
        """
        intent = cls._classify_single(message)
        if intent != "knowledge" or not cls._is_followup(message):
            return intent

        for turn in reversed(cls._recent(history)):
            if turn.role != "user":
                continue
            prior = cls._classify_single(turn.content)
            # Only the most recent user turn is consulted; looking further back
            # reuses intent long after the customer has moved on.
            return prior if prior != "knowledge" else intent
        return intent

    # ------------------------------------------------------------------
    # Identifier extraction (current message first, then recent history)
    # ------------------------------------------------------------------

    @classmethod
    def _texts(
        cls, message: str, history: Sequence[ChatMessage] | None
    ) -> list[str]:
        """Current message first, then prior USER turns, most recent first.

        Assistant turns are deliberately excluded. They quote example
        identifiers ("for example CUST-1001") and echo prior lookups, so
        scanning them would let the assistant answer a question about an
        account the customer never named.
        """
        return [message] + [
            t.content for t in reversed(cls._recent(history)) if t.role == "user"
        ]

    @classmethod
    def _find_identifier(
        cls,
        pattern: re.Pattern[str],
        message: str,
        history: Sequence[ChatMessage] | None,
    ) -> str | None:
        for text in cls._texts(message, history):
            match = pattern.search(text)
            if match:
                return f"{match.group(1).upper()}-{match.group(2)}"
        return None

    @classmethod
    def _extract_order_id(
        cls, message: str, history: Sequence[ChatMessage] | None = None
    ) -> str | None:
        return cls._find_identifier(_ORDER_RE, message, history)

    @classmethod
    def _extract_customer_id(
        cls, message: str, history: Sequence[ChatMessage] | None = None
    ) -> str | None:
        return cls._find_identifier(_CUSTOMER_RE, message, history)

    @classmethod
    def _extract_location(
        cls, message: str, history: Sequence[ChatMessage] | None = None
    ) -> str | None:
        locations = known_locations()
        for text in cls._texts(message, history):
            lowered = text.lower()
            for location in locations:
                if location.lower() in lowered:
                    return location
        return None

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _envelope(
        result: LLMResult,
        intent: str,
        sources: list[str],
        tool_calls: list[str],
    ) -> dict:
        return {
            "answer": result.text,
            "intent": intent,
            "sources": sources,
            "tool_calls": tool_calls,
            "mode": result.mode,
        }

    @staticmethod
    def _format_context(results: list[dict]) -> str:
        return CONTEXT_SEPARATOR.join(
            f"Source: {r['source']}\n{r['content']}" for r in results
        )

    def _answer_from_knowledge(
        self,
        message: str,
        history: Sequence[ChatMessage] | None,
        intent: str,
        suffix: str = "",
    ) -> dict:
        results = self.retriever.search(message)
        if not results:
            result = self.llm.generate(message, "", suffix, history)
            return self._envelope(result, intent, [], [])

        context = self._format_context(results)
        result = self.llm.generate(message, context, "", history)
        text = f"{result.text}\n\n{suffix}" if suffix else result.text
        return self._envelope(
            LLMResult(text, result.mode),
            intent,
            [r["source"] for r in results],
            [],
        )

    def _answer_from_tool(
        self,
        message: str,
        history: Sequence[ChatMessage] | None,
        intent: str,
        tool_name: str,
        tool_text: str,
    ) -> dict:
        result = self.llm.generate(message, "", tool_text, history)
        return self._envelope(result, intent, [], [tool_name])

    # ------------------------------------------------------------------

    def respond(
        self,
        message: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> dict:
        intent = self.classify(message, history)

        if intent == "order_status":
            order_id = self._extract_order_id(message, history)
            if not order_id:
                # The customer may be asking about order policy rather than a
                # specific order, so answer from the knowledge base and offer
                # the lookup instead of only demanding an order number.
                return self._answer_from_knowledge(
                    message,
                    history,
                    intent,
                    suffix=(
                        "If you want me to check a specific order, please share "
                        "the order number, for example ORD-10234."
                    ),
                )

            result = get_order_status(order_id)
            logger.info(
                "Tool invoked",
                extra={
                    "tool": "get_order_status",
                    "order_id": order_id,
                    "found": result["found"],
                },
            )
            if result["found"]:
                tool_text = (
                    f"Order {result['order_id']} is currently {result['status']}. "
                    f"{result['eta']}."
                )
            else:
                tool_text = (
                    f"I could not find order {result['order_id']} in the demo "
                    "order system."
                )
            return self._answer_from_tool(
                message, history, intent, "get_order_status", tool_text
            )

        if intent == "billing":
            customer_id = self._extract_customer_id(message, history)
            if not customer_id:
                return self._answer_from_knowledge(
                    message,
                    history,
                    intent,
                    suffix=(
                        "If you want me to check your account specifically, "
                        "please share your customer ID, for example CUST-1001."
                    ),
                )

            result = get_billing_status(customer_id)
            logger.info(
                "Tool invoked",
                extra={
                    "tool": "get_billing_status",
                    "customer_id": customer_id,
                    "found": result["found"],
                },
            )
            if result["found"]:
                tool_text = (
                    f"Customer {result['customer_id']} has a balance of "
                    f"{result['balance']}, due {result['due_date']}."
                )
            else:
                tool_text = (
                    f"I could not find customer {result['customer_id']} in the "
                    "demo billing system."
                )
            return self._answer_from_tool(
                message, history, intent, "get_billing_status", tool_text
            )

        if intent == "service_status":
            location = self._extract_location(message, history)
            if not location:
                return self._answer_from_knowledge(
                    message,
                    history,
                    intent,
                    suffix=(
                        "Tell me which location you want me to check and I will "
                        "look up the current service status."
                    ),
                )

            result = get_service_status(location)
            logger.info(
                "Tool invoked",
                extra={
                    "tool": "get_service_status",
                    "location": location,
                    "found": result["found"],
                },
            )
            return self._answer_from_tool(
                message, history, intent, "get_service_status", result["status"]
            )

        return self._answer_from_knowledge(message, history, intent)
