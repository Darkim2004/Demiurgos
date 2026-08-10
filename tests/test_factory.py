"""Tests for environment-based provider selection."""

import pytest

from models.factory import create_provider
from models.groq_provider import GroqProvider
from models.ollama_provider import OllamaProvider


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    provider = create_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3:8b"


def test_factory_creates_the_configured_ollama_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "OLLAMA")
    monkeypatch.setenv("MODEL", "custom-model")

    provider = create_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "custom-model"


def test_factory_creates_the_configured_groq_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("MODEL", "groq-model")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    provider = create_provider()

    assert isinstance(provider, GroqProvider)
    assert provider.model == "groq-model"


@pytest.mark.parametrize("missing_name", ["MODEL", "GROQ_API_KEY"])
def test_factory_requires_groq_configuration(monkeypatch, missing_name):
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("MODEL", "groq-model")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv(missing_name)

    with pytest.raises(ValueError, match=missing_name):
        create_provider()


def test_factory_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unknown model provider: unknown"):
        create_provider()
