"""Provider registry and factory.

Register a provider with @register_provider("name"); build one with
build_provider(settings). New providers are added without touching callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from app.core import get_logger
from app.llm.base import LLMProvider

if TYPE_CHECKING:
    from app.config import Settings

logger = get_logger(__name__)

_REGISTRY: dict[str, Callable[[Settings], LLMProvider]] = {}


def register_provider(name: str) -> Callable[[Callable[[Settings], LLMProvider]], Callable[[Settings], LLMProvider]]:
    """Decorator registering a builder under a provider name."""

    def decorator(builder: Callable[[Settings], LLMProvider]) -> Callable[[Settings], LLMProvider]:
        _REGISTRY[name] = builder
        return builder

    return decorator


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider selected in settings.llm_provider."""
    builder = _REGISTRY.get(settings.llm_provider)
    if builder is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown llm_provider '{settings.llm_provider}'. Known: {known}")
    logger.info("Using LLM provider: %s", settings.llm_provider)
    return builder(settings)


def available_providers() -> list[str]:
    """Return the names of all registered providers."""
    return sorted(_REGISTRY)
