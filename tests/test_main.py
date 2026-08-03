"""Tests for the command-line interface and Ollama boundary."""

import os
from types import SimpleNamespace

import pytest

import main


def test_run_agent_sends_the_message_to_the_agent(monkeypatch):
    """The agent response is returned and the Ollama request is correct."""
    agent_response = SimpleNamespace(
        message=SimpleNamespace(content="Hello from the test agent!")
    )
    chat_calls = []

    def fake_chat(**kwargs):
        chat_calls.append(kwargs)
        return agent_response

    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setattr(main.ollama, "chat", fake_chat)

    response = main.run_agent("Hello")

    assert response == "Hello from the test agent!"
    assert chat_calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    ]


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set RUN_OLLAMA_INTEGRATION=1 to run against a live Ollama agent.",
)
def test_run_agent_with_a_live_agent():
    """Optional integration test; requires Ollama and the configured model."""
    response = main.run_agent("Reply with a short greeting.")

    assert isinstance(response, str)
    assert response.strip()


def test_main_runs_without_a_live_agent(monkeypatch, capsys):
    """The CLI can be tested offline by replacing the agent boundary."""
    prompts = []
    agent_messages = []

    def fake_input(prompt):
        prompts.append(prompt)
        return "Hello"

    def fake_run_agent(message):
        agent_messages.append(message)
        return "Stubbed response"

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(main, "run_agent", fake_run_agent)

    main.main()

    assert prompts == ["Enter a message: "]
    assert agent_messages == ["Hello"]
    assert capsys.readouterr().out == (
        "Hello from demiurgos!\nAgent response: Stubbed response\n"
    )
