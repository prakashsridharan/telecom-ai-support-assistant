"""Provider selection and the demo responder."""

from app.services.providers import build_provider, provider_status
from app.services.providers.base import demo_text
from app.services.providers.demo import DemoProvider


def test_demo_provider_is_the_default(monkeypatch):
    provider = build_provider()
    assert provider.name == "demo"
    assert not provider.supports_tool_calling


def test_provider_without_a_key_degrades_to_demo(monkeypatch):
    """Selecting a provider but omitting its key must not crash the app."""
    monkeypatch.setattr("app.services.providers.settings.llm_provider", "anthropic")
    monkeypatch.setattr("app.services.providers.settings.anthropic_api_key", "")

    assert build_provider().name == "demo"

    status, detail = provider_status()
    assert status == "degraded"
    assert "ANTHROPIC_API_KEY" in detail


def test_unknown_provider_degrades_to_demo(monkeypatch):
    monkeypatch.setattr("app.services.providers.settings.llm_provider", "llama")

    assert build_provider().name == "demo"
    status, detail = provider_status()
    assert status == "degraded"
    assert "llama" in detail


def test_configured_provider_is_reported_ready(monkeypatch):
    monkeypatch.setattr("app.services.providers.settings.llm_provider", "anthropic")
    monkeypatch.setattr("app.services.providers.settings.anthropic_api_key", "sk-test")
    monkeypatch.setattr(
        "app.services.providers.settings.anthropic_model", "claude-opus-5"
    )

    status, detail = provider_status()
    assert status == "ok"
    assert "claude-opus-5" in detail


def test_demo_provider_passes_tool_results_through():
    result = DemoProvider().generate("x", "", "Order ORD-1 is currently Shipped.")
    assert result.mode == "demo"
    assert result.text == "Order ORD-1 is currently Shipped."


def test_demo_provider_shows_the_whole_top_document():
    context = "Source: plans.md\n# Telecom Plans\n\n## Premium\n- Monthly price: $79"
    assert "$79" in DemoProvider().generate("premium price", context, "").text


def test_demo_provider_refuses_without_evidence():
    assert "don't have enough information" in demo_text("capital of Peru", "", "")


def test_demo_provider_cannot_run_an_agent():
    import pytest

    with pytest.raises(NotImplementedError):
        DemoProvider().run_agent("hi", None, None)
