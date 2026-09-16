"""The two routing strategies, through the orchestrator."""

from app.services.orchestrator import Orchestrator
from app.services.providers.anthropic_provider import AnthropicProvider
from tests.fakes import (
    FakeAnthropicClient,
    anthropic_text_turn,
    anthropic_tool_turn,
)


def _agent_orchestrator(monkeypatch, responses):
    monkeypatch.setattr(
        "app.services.orchestrator.settings.routing_mode", "agent"
    )
    provider = AnthropicProvider(client=FakeAnthropicClient(responses))
    return Orchestrator(provider=provider)


def test_rules_mode_is_the_default():
    orchestrator = Orchestrator()
    assert orchestrator.routing_mode == "rules"
    assert orchestrator.respond("Check order ORD-10234")["routing"] == "rules"


def test_agent_mode_degrades_to_rules_without_a_capable_provider(monkeypatch):
    """Asking for agent mode in demo mode must not break the app."""
    monkeypatch.setattr(
        "app.services.orchestrator.settings.routing_mode", "agent"
    )
    orchestrator = Orchestrator()  # demo provider

    assert orchestrator.routing_mode == "rules"
    result = orchestrator.respond("Check order ORD-10234")
    assert result["routing"] == "rules"
    assert result["tool_calls"] == ["get_order_status"]


def test_agent_mode_lets_the_model_choose_the_tool(monkeypatch):
    orchestrator = _agent_orchestrator(
        monkeypatch,
        [
            anthropic_tool_turn("look_up_order", {"order_id": "ORD-10234"}),
            anthropic_text_turn("Your order has shipped, arriving 2026-09-10."),
        ],
    )

    result = orchestrator.respond("any news on ORD-10234")

    assert result["routing"] == "agent"
    assert result["mode"] == "agent"
    assert result["tool_calls"] == ["look_up_order"]
    assert result["provider"] == "anthropic"


def test_agent_mode_can_answer_without_calling_any_tool(monkeypatch):
    orchestrator = _agent_orchestrator(
        monkeypatch, [anthropic_text_turn("Hello, how can I help?")]
    )

    result = orchestrator.respond("hello")

    assert result["tool_calls"] == []
    assert result["sources"] == []


def test_agent_mode_reports_knowledge_sources(monkeypatch):
    orchestrator = _agent_orchestrator(
        monkeypatch,
        [
            anthropic_tool_turn("search_knowledge_base", {"query": "plans"}),
            anthropic_text_turn("We offer Standard, Premium and Business."),
        ],
    )

    result = orchestrator.respond("what plans do you offer?")

    assert result["sources"] == ["plans.md"]
    assert result["tool_calls"] == ["search_knowledge_base"]


def test_agent_mode_still_reports_an_intent(monkeypatch):
    """Intent is observed, not used for routing — it keeps both strategies
    comparable in the evaluation harness."""
    orchestrator = _agent_orchestrator(
        monkeypatch,
        [
            anthropic_tool_turn("look_up_order", {"order_id": "ORD-10234"}),
            anthropic_text_turn("Shipped."),
        ],
    )

    assert orchestrator.respond("Check order ORD-10234")["intent"] == "order_status"


def test_response_envelope_is_identical_across_modes(monkeypatch):
    rules = Orchestrator().respond("Check order ORD-10234")
    agent = _agent_orchestrator(
        monkeypatch,
        [
            anthropic_tool_turn("look_up_order", {"order_id": "ORD-10234"}),
            anthropic_text_turn("Shipped."),
        ],
    ).respond("Check order ORD-10234")

    assert set(rules) == set(agent)
