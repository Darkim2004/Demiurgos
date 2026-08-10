"""Ollama adapter for the provider-independent model interface."""

from typing import Any, Dict, Sequence

import ollama

from models.base import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class OllamaProvider(ModelProvider):
    """Translate between Demiurgos types and the Ollama Python client."""

    def __init__(self, model: str):
        self.model = model

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> ModelResponse:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
        }
        if tools:
            request["tools"] = [self._convert_tool(tool) for tool in tools]

        response = ollama.chat(**request)
        normalized_tool_calls = [
            ToolCall(
                id=f"ollama-tool-call-{index}",
                name=tool_call.function.name,
                arguments=dict(tool_call.function.arguments),
            )
            for index, tool_call in enumerate(response.message.tool_calls or ())
        ]

        return ModelResponse(
            content=response.message.content,
            tool_calls=normalized_tool_calls,
        )

    @staticmethod
    def _convert_message(message: ChatMessage) -> Dict[str, Any]:
        converted: Dict[str, Any] = {"role": message.role}
        if message.content is not None:
            converted["content"] = message.content
        if message.tool_calls:
            converted["tool_calls"] = [
                {
                    "function": {
                        "name": tool_call.name,
                        "arguments": dict(tool_call.arguments),
                    }
                }
                for tool_call in message.tool_calls
            ]
        if message.role == "tool" and message.name:
            converted["tool_name"] = message.name
        return converted

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
