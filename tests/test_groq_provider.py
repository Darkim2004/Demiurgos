"""Offline tests for the Groq adapter."""

import json
from types import SimpleNamespace

import pytest

from agent import ADD_TOOL
from models.base import ChatMessage, ToolCall
from models.groq_provider import GroqProvider


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return self.response


def test_groq_provider_normalizes_a_text_response():
    native_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from Groq",
                    tool_calls=None,
                )
            )
        ]
    )
    completions = FakeCompletions(native_response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response = GroqProvider("test-model", "test-key", client=client).chat(
        [ChatMessage(role="user", content="Hello")]
    )

    assert response.content == "Hello from Groq"
    assert response.tool_calls == []
    assert completions.requests == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    ]


def test_groq_provider_translates_messages_tools_and_response():
    native_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="groq-call-1",
                            function=SimpleNamespace(
                                name="add",
                                arguments='{"a": 2, "b": 3}',
                            ),
                        )
                    ],
                )
            )
        ]
    )
    completions = FakeCompletions(native_response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = GroqProvider("test-model", "test-key", client=client)

    response = provider.chat(
        [
            ChatMessage(role="user", content="Add two and three"),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall("groq-call-1", "add", {"a": 2, "b": 3})],
            ),
            ChatMessage(
                role="tool",
                content="5",
                tool_call_id="groq-call-1",
                name="add",
            ),
        ],
        [ADD_TOOL.definition],
    )

    assert response.content is None
    assert response.tool_calls == [
        ToolCall("groq-call-1", "add", {"a": 2, "b": 3})
    ]
    request = completions.requests[0]
    assert request["model"] == "test-model"
    assert request["messages"][0] == {
        "role": "user",
        "content": "Add two and three",
    }
    assert request["messages"][1]["role"] == "assistant"
    assert request["messages"][1]["tool_calls"][0]["id"] == "groq-call-1"
    assert json.loads(
        request["messages"][1]["tool_calls"][0]["function"]["arguments"]
    ) == {"a": 2, "b": 3}
    assert request["messages"][2] == {
        "role": "tool",
        "content": "5",
        "tool_call_id": "groq-call-1",
        "name": "add",
    }
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": ADD_TOOL.definition.parameters,
            },
        }
    ]


def test_groq_provider_rejects_a_tool_message_without_an_id():
    provider = GroqProvider(
        "test-model",
        "test-key",
        client=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="tool_call_id"):
        provider.chat([ChatMessage(role="tool", content="5", name="add")])
