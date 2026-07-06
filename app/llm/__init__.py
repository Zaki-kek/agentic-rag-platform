"""LLM provider package.

Importing this package registers all built-in providers via their decorators.
"""

# Import provider modules for their registration side effects.
from app.llm import anthropic_provider, echo, gigachat, openai_provider  # noqa: E402,F401
from app.llm.base import LLMProvider, Message
from app.llm.factory import available_providers, build_provider, register_provider

__all__ = [
    "LLMProvider",
    "Message",
    "build_provider",
    "available_providers",
    "register_provider",
]
