"""Provider-neutral tool definitions and dispatch.

One source of truth for what the assistant can *do*. Both the Anthropic and
OpenAI providers render these same specs into their own wire format, and the
rule-based orchestrator calls the same underlying functions — so switching
provider or routing strategy never changes the capability surface.

Tool descriptions matter as much as the code here: in agent mode they are the
only thing telling the model when a lookup is appropriate. They are written to
discourage guessing at identifiers.
"""

import logging
from dataclasses import dataclass, field

from app.tools.telecom_tools import (
    get_billing_status,
    get_order_status,
    get_service_status,
    known_locations,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """What a tool produced, plus where it came from."""

    text: str
    found: bool = True
    sources: list[str] = field(default_factory=list)


def _locations_hint() -> str:
    return ", ".join(known_locations())


# Provider-neutral specs. `input_schema` is JSON Schema; each provider adapts
# the envelope (Anthropic: input_schema, OpenAI: function.parameters).
TOOL_SPECS: list[dict] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the telecom knowledge base for plans, pricing, billing "
            "policy, order policy and service policy. Use this for any general "
            "question about what the company offers or how its policies work. "
            "Returns the matching policy documents, or nothing if the knowledge "
            "base does not cover the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question, in their own words.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "look_up_order",
        "description": (
            "Look up the current status and delivery estimate of a specific "
            "order. Requires the customer's order number. Never invent or guess "
            "an order number: if the customer has not given one, ask for it "
            "instead of calling this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order number, for example ORD-10234.",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "look_up_billing",
        "description": (
            "Look up the outstanding balance and due date for a specific "
            "customer account. Requires the customer ID. Never invent or guess "
            "a customer ID: if the customer has not given one, ask for it "
            "instead of calling this tool. For general billing policy questions "
            "use search_knowledge_base instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID, for example CUST-1001.",
                }
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_service_status",
        "description": (
            "Check for outages or planned maintenance in a named location. "
            f"Known locations: {_locations_hint()}. Use this only when the "
            "customer reports a network problem or asks about an outage in a "
            "specific place. Do not claim a service is healthy without calling "
            "this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, for example Chennai.",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = frozenset(spec["name"] for spec in TOOL_SPECS)


class ToolBox:
    """Executes tool calls against the synthetic business systems.

    Holds the retriever so `search_knowledge_base` is a tool like any other —
    which is what lets agent mode treat retrieval as a decision the model makes
    rather than a step the router hard-codes.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def execute(self, name: str, arguments: dict) -> ToolResult:
        handler = {
            "search_knowledge_base": self._search_knowledge_base,
            "look_up_order": self._look_up_order,
            "look_up_billing": self._look_up_billing,
            "check_service_status": self._check_service_status,
        }.get(name)

        if handler is None:
            logger.warning("Unknown tool requested", extra={"tool": name})
            return ToolResult(f"Unknown tool: {name}", found=False)

        try:
            result = handler(arguments)
        except Exception:
            # A tool raising must not end the conversation. The model receives
            # the failure as a tool result and can tell the customer.
            logger.exception("Tool raised", extra={"tool": name})
            return ToolResult(
                f"The {name} system is temporarily unavailable.", found=False
            )

        logger.info(
            "Tool invoked",
            extra={"tool": name, "arguments": arguments, "found": result.found},
        )
        return result

    # ------------------------------------------------------------------

    def _search_knowledge_base(self, arguments: dict) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult("No query supplied.", found=False)

        results = self.retriever.search(query)
        if not results:
            return ToolResult(
                "The knowledge base contains nothing relevant to that question.",
                found=False,
            )

        text = "\n\n-----\n\n".join(
            f"Source: {r['source']}\n{r['content']}" for r in results
        )
        return ToolResult(text, found=True, sources=[r["source"] for r in results])

    def _look_up_order(self, arguments: dict) -> ToolResult:
        order_id = str(arguments.get("order_id", "")).strip().upper()
        if not order_id:
            return ToolResult("No order number supplied.", found=False)

        result = get_order_status(order_id)
        if not result["found"]:
            return ToolResult(
                f"No order named {result['order_id']} exists in the order "
                "system. Do not describe a status for it.",
                found=False,
            )
        return ToolResult(
            f"Order {result['order_id']} is currently {result['status']}. "
            f"{result['eta']}."
        )

    def _look_up_billing(self, arguments: dict) -> ToolResult:
        customer_id = str(arguments.get("customer_id", "")).strip().upper()
        if not customer_id:
            return ToolResult("No customer ID supplied.", found=False)

        result = get_billing_status(customer_id)
        if not result["found"]:
            return ToolResult(
                f"No customer named {result['customer_id']} exists in the "
                "billing system. Do not describe a balance for it.",
                found=False,
            )
        return ToolResult(
            f"Customer {result['customer_id']} has a balance of "
            f"{result['balance']}, due {result['due_date']}."
        )

    def _check_service_status(self, arguments: dict) -> ToolResult:
        location = str(arguments.get("location", "")).strip()
        if not location:
            return ToolResult("No location supplied.", found=False)

        result = get_service_status(location)
        return ToolResult(result["status"], found=result["found"])
