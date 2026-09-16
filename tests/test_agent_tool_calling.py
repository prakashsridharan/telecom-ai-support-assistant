"""Real tool calling, exercised without an API key.

These cover the agent loop itself: that tool schemas reach the provider, that
the model's chosen arguments are dispatched to the right function, that results
are framed correctly for the next turn, and that the loop terminates.
"""

import pytest

from app.models import ChatMessage
from app.services.providers.anthropic_provider import AnthropicProvider
from app.services.providers.openai_provider import OpenAIProvider
from app.services.retriever import KnowledgeRetriever
from app.services.tools import TOOL_NAMES, ToolBox
from tests.fakes import (
    BoomAnthropicClient,
    FakeAnthropicClient,
    FakeAnthropicResponse,
    FakeOpenAIClient,
    anthropic_text_turn,
    anthropic_tool_turn,
    openai_text_turn,
    openai_tool_turn,
)


@pytest.fixture
def toolbox():
    return ToolBox(KnowledgeRetriever())


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------


def test_claude_calls_the_order_tool_and_answers(toolbox):
    client = FakeAnthropicClient(
        [
            anthropic_tool_turn("look_up_order", {"order_id": "ORD-10234"}),
            anthropic_text_turn("Your order ORD-10234 has shipped."),
        ]
    )
    provider = AnthropicProvider(client=client)

    result = provider.run_agent("Where is ORD-10234?", None, toolbox)

    assert result.mode == "agent"
    assert result.tool_calls == ["look_up_order"]
    assert "shipped" in result.text.lower()


def test_tool_result_is_fed_back_to_the_model(toolbox):
    client = FakeAnthropicClient(
        [
            anthropic_tool_turn("look_up_order", {"order_id": "ORD-10234"}),
            anthropic_text_turn("Done."),
        ]
    )
    AnthropicProvider(client=client).run_agent("status?", None, toolbox)

    # Second request carries the assistant turn plus a tool_result referencing
    # the same tool_use id.
    second = client.requests[1]["messages"]
    tool_results = [
        block
        for entry in second
        if isinstance(entry.get("content"), list)
        for block in entry["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "toolu_1"
    assert "Shipped" in tool_results[0]["content"]
    assert tool_results[0]["is_error"] is False


def test_all_tools_are_offered_to_the_model(toolbox):
    client = FakeAnthropicClient([anthropic_text_turn("Hello.")])
    AnthropicProvider(client=client).run_agent("hi", None, toolbox)

    offered = {tool["name"] for tool in client.requests[0]["tools"]}
    assert offered == set(TOOL_NAMES)
    # strict guarantees the arguments validate against the schema.
    assert all(tool["strict"] for tool in client.requests[0]["tools"])


def test_unknown_record_is_flagged_as_an_error_result(toolbox):
    client = FakeAnthropicClient(
        [
            anthropic_tool_turn("look_up_order", {"order_id": "ORD-99999"}),
            anthropic_text_turn("I couldn't find that order."),
        ]
    )
    AnthropicProvider(client=client).run_agent("ORD-99999?", None, toolbox)

    second = client.requests[1]["messages"]
    result_block = next(
        block
        for entry in second
        if isinstance(entry.get("content"), list)
        for block in entry["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    assert result_block["is_error"] is True
    assert "No order named" in result_block["content"]


def test_knowledge_search_tool_reports_its_sources(toolbox):
    client = FakeAnthropicClient(
        [
            anthropic_tool_turn(
                "search_knowledge_base", {"query": "premium plan price"}
            ),
            anthropic_text_turn("The Premium plan is $79 per month."),
        ]
    )
    result = AnthropicProvider(client=client).run_agent("premium?", None, toolbox)

    assert result.tool_calls == ["search_knowledge_base"]
    assert "plans.md" in result.sources


def test_multiple_tool_calls_in_one_turn(toolbox):
    """Parallel calls must all come back in a single user message."""
    from tests.fakes import FakeToolUseBlock

    client = FakeAnthropicClient(
        [
            FakeAnthropicResponse(
                content=[
                    FakeToolUseBlock("look_up_order", {"order_id": "ORD-10234"}, "a"),
                    FakeToolUseBlock("look_up_billing", {"customer_id": "CUST-1001"}, "b"),
                ],
                stop_reason="tool_use",
            ),
            anthropic_text_turn("Both done."),
        ]
    )
    result = AnthropicProvider(client=client).run_agent("both?", None, toolbox)

    assert result.tool_calls == ["look_up_order", "look_up_billing"]
    tool_result_messages = [
        entry
        for entry in client.requests[1]["messages"]
        if entry["role"] == "user"
        and isinstance(entry["content"], list)
        and all(b.get("type") == "tool_result" for b in entry["content"])
    ]
    assert len(tool_result_messages) == 1
    assert len(tool_result_messages[0]["content"]) == 2


def test_iteration_cap_stops_a_runaway_loop(toolbox):
    """A model that never stops calling tools must still return."""
    client = FakeAnthropicClient(
        [anthropic_tool_turn("look_up_order", {"order_id": "ORD-10234"})] * 20
    )
    result = AnthropicProvider(client=client).run_agent("loop", None, toolbox)

    assert result.mode == "fallback"
    assert len(client.requests) <= 6


def test_provider_failure_degrades_to_retrieval(toolbox):
    result = AnthropicProvider(client=BoomAnthropicClient()).run_agent(
        "What plans do you offer?", None, toolbox
    )
    assert result.mode == "fallback"
    assert "plans.md" in result.sources
    assert "$79" in result.text


def test_refusal_is_handled_not_crashed(toolbox):
    client = FakeAnthropicClient(
        [FakeAnthropicResponse(content=[], stop_reason="refusal")]
    )
    result = AnthropicProvider(client=client).run_agent("...", None, toolbox)
    assert "not able to help" in result.text.lower()


def test_history_is_passed_to_the_model(toolbox):
    client = FakeAnthropicClient([anthropic_text_turn("Still shipped.")])
    history = [
        ChatMessage(role="user", content="Check ORD-10234"),
        ChatMessage(role="assistant", content="It has shipped."),
    ]
    AnthropicProvider(client=client).run_agent("Any update?", history, toolbox)

    sent = client.requests[0]["messages"]
    assert sent[0]["content"] == "Check ORD-10234"
    assert sent[-1]["content"] == "Any update?"


# ----------------------------------------------------------------------
# OpenAI — same contract, different wire format
# ----------------------------------------------------------------------


def test_openai_calls_tools_and_answers(toolbox):
    client = FakeOpenAIClient(
        [
            openai_tool_turn("look_up_billing", '{"customer_id": "CUST-1002"}'),
            openai_text_turn("Your balance is $79.00."),
        ]
    )
    result = OpenAIProvider(client=client).run_agent("balance?", None, toolbox)

    assert result.mode == "agent"
    assert result.tool_calls == ["look_up_billing"]
    assert "$79.00" in result.text


def test_openai_tool_result_is_fed_back(toolbox):
    client = FakeOpenAIClient(
        [
            openai_tool_turn("look_up_billing", '{"customer_id": "CUST-1002"}'),
            openai_text_turn("Done."),
        ]
    )
    OpenAIProvider(client=client).run_agent("balance?", None, toolbox)

    tool_messages = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "$79.00" in tool_messages[0]["content"]


def test_openai_malformed_arguments_do_not_crash(toolbox):
    client = FakeOpenAIClient(
        [
            openai_tool_turn("look_up_order", "{not valid json"),
            openai_text_turn("I need an order number."),
        ]
    )
    result = OpenAIProvider(client=client).run_agent("status?", None, toolbox)
    assert result.mode == "agent"
    assert result.tool_calls == ["look_up_order"]
