"""Offline tests for the Ollama adapter."""

from types import SimpleNamespace

from agent import ADD_TOOL
from models.base import ChatMessage, ToolCall
from models.ollama_provider import OllamaProvider


def test_ollama_provider_normalizes_a_text_response(monkeypatch):
    requests = []

    def fake_chat(**request):
        requests.append(request)
        return SimpleNamespace(
            message=SimpleNamespace(content="Hello from Ollama", tool_calls=None)
        )

    monkeypatch.setattr("models.ollama_provider.ollama.chat", fake_chat)

    response = OllamaProvider(model="test-model").chat(
        [ChatMessage(role="user", content="Hello")]
    )

    assert response.content == "Hello from Ollama"
    assert response.tool_calls == []
    assert requests == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    ]


def test_ollama_provider_translates_messages_tools_and_response(monkeypatch):
    requests = []
    native_response = SimpleNamespace(
        message=SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="add",
                        arguments={"a": 2, "b": 3},
                    )
                )
            ],
        )
    )

    def fake_chat(**request):
        requests.append(request)
        return native_response

    monkeypatch.setattr("models.ollama_provider.ollama.chat", fake_chat)
    provider = OllamaProvider(model="test-model")

    response = provider.chat(
        [
            ChatMessage(role="user", content="Add two and three"),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall("original-id", "add", {"a": 2, "b": 3})],
            ),
            ChatMessage(
                role="tool",
                content="5",
                tool_call_id="original-id",
                name="add",
            ),
        ],
        [ADD_TOOL.definition],
    )

    assert response.content is None
    assert response.tool_calls == [
        ToolCall(
            id="ollama-tool-call-0",
            name="add",
            arguments={"a": 2, "b": 3},
        )
    ]
    assert requests == [
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Add two and three"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "add",
                                "arguments": {"a": 2, "b": 3},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_name": "add"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two integers.",
                        "parameters": ADD_TOOL.definition.parameters,
                    },
                }
            ],
        }
    ]
