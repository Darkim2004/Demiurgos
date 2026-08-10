"""Tests for the command-line entry point."""

import main


def test_main_runs_without_a_live_provider(monkeypatch, capsys):
    prompts = []
    provider = object()

    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(main, "create_provider", lambda: provider)

    def fake_input(prompt):
        prompts.append(prompt)
        return "Hello"

    agent_calls = []

    def fake_run_agent(message, selected_provider):
        agent_calls.append((message, selected_provider))
        return "Stubbed response"

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(main, "run_agent", fake_run_agent)

    main.main()

    assert prompts == ["Enter a message: "]
    assert agent_calls == [("Hello", provider)]
    assert capsys.readouterr().out == (
        "Hello from demiurgos!\nAgent response: Stubbed response\n"
    )
