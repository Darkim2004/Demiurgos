"""Tests for the provider-independent agent loop."""

from copy import deepcopy
from typing import List, Sequence, Tuple

import pytest

from agent import ADD_TOOL, run_agent
from models.base import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class FakeProvider(ModelProvider):
    """Return queued responses without accessing a model or network."""

    def __init__(self, responses: Sequence[ModelResponse]):
        self.responses = list(responses)
        self.calls: List[Tuple[List[ChatMessage], List[ToolDefinition]]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> ModelResponse:
        self.calls.append((deepcopy(list(messages)), deepcopy(list(tools))))
        return self.responses.pop(0)


def test_run_agent_returns_a_normal_text_response():
    provider = FakeProvider([ModelResponse(content="Hello from the test agent!")])

    response = run_agent("Hello", provider)

    assert response == "Hello from the test agent!"
    assert provider.calls == [
        (
            [ChatMessage(role="user", content="Hello")],
            [ADD_TOOL.definition],
        )
    ]


def test_run_agent_executes_a_normalized_tool_call():
    tool_call = ToolCall(
        id="call-1",
        name="add",
        arguments={"a": 2, "b": 3},
    )
    provider = FakeProvider(
        [
            ModelResponse(content=None, tool_calls=[tool_call]),
            ModelResponse(content="The result is 5."),
        ]
    )

    response = run_agent("What is two plus three?", provider)

    assert response == "The result is 5."
    assert provider.calls[1][0] == [
        ChatMessage(role="user", content="What is two plus three?"),
        ChatMessage(role="assistant", tool_calls=[tool_call]),
        ChatMessage(
            role="tool",
            content="5",
            tool_call_id="call-1",
            name="add",
        ),
    ]


def test_run_agent_reports_an_unknown_tool_to_the_model():
    tool_call = ToolCall(id="call-2", name="missing", arguments={})
    provider = FakeProvider(
        [
            ModelResponse(content=None, tool_calls=[tool_call]),
            ModelResponse(content="That tool is unavailable."),
        ]
    )

    response = run_agent("Use a missing tool", provider)

    assert response == "That tool is unavailable."
    assert provider.calls[1][0][-1].content == "Error: unknown tool 'missing'"


def test_run_agent_stops_after_the_configured_number_of_steps():
    provider = FakeProvider(
        [ModelResponse(content=None, tool_calls=[ToolCall("1", "add", {})])]
    )

    with pytest.raises(RuntimeError, match="maximum of 1 model steps"):
        run_agent("Keep calling tools", provider, max_steps=1)


def test_run_agent_rejects_a_non_positive_step_limit():
    provider = FakeProvider([])

    with pytest.raises(ValueError, match="at least 1"):
        run_agent("Hello", provider, max_steps=0)
