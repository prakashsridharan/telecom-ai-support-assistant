"""Provider selection.

`LLM_PROVIDER` picks the backend: demo | anthropic | openai. A provider that is
configured but unusable (missing key, SDK not installed) degrades to demo
rather than crashing at import — the app must always start, and `/health`
reports the degradation.
"""

import logging

from app.config import settings
from app.services.providers.base import (
    CONTEXT_SEPARATOR,
    ChatProvider,
    LLMResult,
)
from app.services.providers.demo import DemoProvider

logger = logging.getLogger(__name__)

__all__ = [
    "CONTEXT_SEPARATOR",
    "ChatProvider",
    "LLMResult",
    "DemoProvider",
    "build_provider",
    "provider_status",
]

_KEY_FOR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _configured_key(name: str) -> str:
    return {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
    }.get(name, "")


def provider_status() -> tuple[str, str]:
    """(status, detail) for the health endpoint, without building a client."""
    requested = settings.llm_provider.lower()

    if requested == "demo":
        return "ok", "Demo mode; no LLM provider configured"
    if requested not in _KEY_FOR:
        return "degraded", f"Unknown LLM_PROVIDER '{requested}'; serving demo mode"
    if not _configured_key(requested):
        return (
            "degraded",
            f"LLM_PROVIDER is '{requested}' but {_KEY_FOR[requested]} is not set",
        )
    return "ok", f"Provider '{requested}' using {_model_for(requested)}"


def _model_for(name: str) -> str:
    return {
        "anthropic": settings.anthropic_model,
        "openai": settings.openai_model,
    }.get(name, "deterministic")


def build_provider() -> ChatProvider:
    requested = settings.llm_provider.lower()

    if requested == "demo":
        return DemoProvider()

    if requested not in _KEY_FOR:
        logger.warning(
            "Unknown LLM_PROVIDER; serving demo mode",
            extra={"llm_provider": requested},
        )
        return DemoProvider()

    if not _configured_key(requested):
        logger.warning(
            "Provider selected but its API key is empty; serving demo mode",
            extra={"llm_provider": requested, "expected_env": _KEY_FOR[requested]},
        )
        return DemoProvider()

    try:
        if requested == "anthropic":
            from app.services.providers.anthropic_provider import AnthropicProvider

            provider: ChatProvider = AnthropicProvider()
        else:
            from app.services.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider()
    except Exception:
        # A missing SDK or a malformed key must not stop the service booting.
        logger.exception(
            "Could not construct provider; serving demo mode",
            extra={"llm_provider": requested},
        )
        return DemoProvider()

    logger.info(
        "Provider ready",
        extra={"llm_provider": provider.name, "model": provider.model},
    )
    return provider
