"""Opt-in smoke tests for live model providers."""

import os

import pytest
from dotenv import load_dotenv

from agent import run_agent
from models.groq_provider import GroqProvider
from models.ollama_provider import OllamaProvider


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set RUN_OLLAMA_INTEGRATION=1 to test a live Ollama provider.",
)
def test_run_agent_with_a_live_ollama_provider():
    load_dotenv()
    provider = OllamaProvider(model=os.getenv("MODEL", "qwen3:8b"))

    response = run_agent("Reply with a short greeting.", provider)

    assert response.strip()


@pytest.mark.skipif(
    os.getenv("RUN_GROQ_INTEGRATION") != "1",
    reason="Set RUN_GROQ_INTEGRATION=1 to test a live Groq provider.",
)
def test_run_agent_with_a_live_groq_provider():
    load_dotenv()
    model = os.getenv("MODEL")
    api_key = os.getenv("GROQ_API_KEY")
    if not model or not api_key:
        pytest.fail("MODEL and GROQ_API_KEY are required for the Groq integration test")

    response = run_agent(
        "Reply with a short greeting.",
        GroqProvider(model=model, api_key=api_key),
    )

    assert response.strip()
