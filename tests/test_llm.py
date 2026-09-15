"""LLM service behaviour, including graceful degradation."""

from app.services.llm import LLMService


class _BoomClient:
    """Stands in for an OpenAI client that fails (timeout, 429, auth, outage)."""

    class chat:  # noqa: N801 - mirrors the SDK's attribute layout
        class completions:
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("provider unavailable")


def test_demo_mode_when_no_key_configured():
    service = LLMService()
    assert not service.enabled
    result = service.generate("hello", "", "")
    assert result.mode == "demo"


def test_tool_results_are_passed_through_verbatim():
    result = LLMService().generate("x", "", "Order ORD-1 is currently Shipped.")
    assert result.text == "Order ORD-1 is currently Shipped."


def test_demo_mode_shows_the_whole_top_document_not_just_its_heading():
    context = "Source: plans.md\n# Telecom Plans\n\n## Premium\n- Monthly price: $79"
    result = LLMService().generate("premium price", context, "")
    assert "$79" in result.text


def test_refuses_when_there_is_no_evidence():
    result = LLMService().generate("capital of Peru", "", "")
    assert "don't have enough information" in result.text


def test_provider_failure_degrades_to_demo_instead_of_raising():
    service = LLMService()
    service.enabled = True
    service.client = _BoomClient()

    result = service.generate("x", "", "Order ORD-1 is currently Shipped.")

    # The customer still gets a grounded answer, and the response is labelled
    # so the degradation is visible rather than silent.
    assert result.mode == "fallback"
    assert result.text == "Order ORD-1 is currently Shipped."
