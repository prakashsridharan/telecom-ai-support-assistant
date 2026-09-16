"""Fake provider clients that replay scripted responses.

Tool-calling code is the easiest part of an LLM app to leave untested, because
the obvious way to exercise it costs money and is non-deterministic. These
stand-ins replay canned turns with the same shape the SDKs return, so the loop
itself — argument parsing, tool dispatch, result framing, iteration caps,
failure handling — is covered by ordinary unit tests.
"""

from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Anthropic shapes
# ----------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class FakeAnthropicResponse:
    content: list
    stop_reason: str = "end_turn"
    stop_details: object = None


class FakeAnthropicClient:
    """Replays a scripted list of responses and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.requests.append(kwargs)
            if not self._outer._responses:
                raise AssertionError("FakeAnthropicClient ran out of responses")
            return self._outer._responses.pop(0)


class BoomAnthropicClient:
    """Every call fails — a timeout, 429, auth error or provider outage."""

    def __init__(self):
        self.messages = self._Messages()

    class _Messages:
        @staticmethod
        def create(**_kwargs):
            raise RuntimeError("provider unavailable")


# ----------------------------------------------------------------------
# OpenAI shapes
# ----------------------------------------------------------------------


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeOpenAIToolCall:
    function: FakeFunction
    id: str = "call_1"
    type: str = "function"


@dataclass
class FakeOpenAIMessage:
    content: str | None = None
    tool_calls: list = field(default_factory=list)


@dataclass
class FakeChoice:
    message: FakeOpenAIMessage


@dataclass
class FakeOpenAIResponse:
    choices: list


class FakeOpenAIClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer):
            self.completions = FakeOpenAIClient._Completions(outer)

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.requests.append(kwargs)
            if not self._outer._responses:
                raise AssertionError("FakeOpenAIClient ran out of responses")
            return self._outer._responses.pop(0)


def anthropic_tool_turn(name: str, arguments: dict, block_id: str = "toolu_1"):
    """One assistant turn that calls a tool."""
    return FakeAnthropicResponse(
        content=[FakeToolUseBlock(name=name, input=arguments, id=block_id)],
        stop_reason="tool_use",
    )


def anthropic_text_turn(text: str):
    """One assistant turn that answers."""
    return FakeAnthropicResponse(content=[FakeTextBlock(text=text)])


def openai_tool_turn(name: str, arguments: str, call_id: str = "call_1"):
    return FakeOpenAIResponse(
        choices=[
            FakeChoice(
                FakeOpenAIMessage(
                    content=None,
                    tool_calls=[
                        FakeOpenAIToolCall(
                            function=FakeFunction(name=name, arguments=arguments),
                            id=call_id,
                        )
                    ],
                )
            )
        ]
    )


def openai_text_turn(text: str):
    return FakeOpenAIResponse(choices=[FakeChoice(FakeOpenAIMessage(content=text))])
