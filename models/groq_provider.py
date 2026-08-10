"""Groq adapter for the provider-independent model interface."""

import json
from typing import Any, Dict, Optional, Sequence

from groq import Groq

from models.base import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class GroqProvider(ModelProvider):
    """Translate between Demiurgos types and the Groq Python client."""

    def __init__(self, model: str, api_key: str, client: Optional[Any] = None):
        self.model = model
        self._client = client if client is not None else Groq(api_key=api_key)

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

        response = self._client.chat.completions.create(**request)
        if not response.choices:
            raise ValueError("Groq returned a response without any choices")

        message = response.choices[0].message
        normalized_tool_calls = [
            self._convert_tool_call(tool_call)
            for tool_call in message.tool_calls or ()
        ]
        return ModelResponse(
            content=message.content,
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
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in message.tool_calls
            ]
        if message.role == "tool":
            if not message.tool_call_id:
                raise ValueError("Groq tool messages require a tool_call_id")
            converted["tool_call_id"] = message.tool_call_id
            if message.name:
                converted["name"] = message.name
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

    @staticmethod
    def _convert_tool_call(tool_call: Any) -> ToolCall:
        raw_arguments = tool_call.function.arguments or "{}"
        arguments = (
            raw_arguments
            if isinstance(raw_arguments, dict)
            else json.loads(raw_arguments)
        )
        if not isinstance(arguments, dict):
            raise ValueError("Groq tool-call arguments must decode to an object")

        return ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=arguments,
        )
