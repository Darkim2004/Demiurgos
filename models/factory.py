"""Create the configured model provider at the application boundary."""

import os

from models.base import ModelProvider
from models.groq_provider import GroqProvider
from models.ollama_provider import OllamaProvider


def create_provider() -> ModelProvider:
    """Build the provider selected through environment variables."""

    provider_name = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()

    if provider_name == "ollama":
        return OllamaProvider(model=os.getenv("MODEL", "qwen3:8b"))

    if provider_name == "groq":
        model = _required_environment_variable("MODEL")
        api_key = _required_environment_variable("GROQ_API_KEY")
        return GroqProvider(model=model, api_key=api_key)

    raise ValueError(f"Unknown model provider: {provider_name}")


def _required_environment_variable(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required for the selected model provider")
    return value
