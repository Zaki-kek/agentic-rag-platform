"""Contract tests: factory-built components satisfy their protocols."""

from __future__ import annotations

from app.config import Settings
from app.llm import available_providers, build_provider
from app.llm.base import LLMProvider
from app.rag.embed import Embedder, build_embedder
from app.rag.store import build_store


def test_echo_provider_registered_and_conforms() -> None:
    assert "echo" in available_providers()
    provider = build_provider(Settings(llm_provider="echo"))
    assert isinstance(provider, LLMProvider)  # runtime_checkable Protocol
    assert provider.name == "echo"


def test_hash_embedder_conforms() -> None:
    embedder = build_embedder(Settings(embedder="hash"))
    assert isinstance(embedder, Embedder)
    assert embedder.dim > 0


def test_memory_store_implements_protocol_methods() -> None:
    store = build_store(Settings(vector_store="memory"), dim=16)
    for method in ("init", "add", "search", "close"):
        assert callable(getattr(store, method))


def test_unknown_provider_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_provider(Settings(llm_provider="does-not-exist"))
