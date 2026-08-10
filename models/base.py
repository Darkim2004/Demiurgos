"""Demiurgos-owned types shared by the agent and model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ToolCall:
    """A provider-independent request to execute a tool."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """The normalized response returned by every model provider."""

    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class ChatMessage:
    """A provider-independent message in an agent conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ToolDefinition:
    """The name and JSON schema exposed to a model for one tool."""

    name: str
    description: str
    parameters: Dict[str, Any]


class ModelProvider(ABC):
    """Interface implemented by every model backend."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> ModelResponse:
        """Return one normalized response for the supplied conversation."""

        raise NotImplementedError
