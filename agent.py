"""Provider-independent agent and local tool execution loop."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

from models.base import ChatMessage, ModelProvider, ToolDefinition
from tools import add


ToolFunction = Callable[..., Any]


@dataclass
class AgentTool:
    """Pair a model-visible tool definition with its local implementation."""

    definition: ToolDefinition
    function: ToolFunction


ADD_TOOL = AgentTool(
    definition=ToolDefinition(
        name="add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First integer."},
                "b": {"type": "integer", "description": "Second integer."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    ),
    function=add,
)

DEFAULT_TOOLS: Tuple[AgentTool, ...] = (ADD_TOOL,)


def run_agent(
    message: str,
    provider: ModelProvider,
    tools: Sequence[AgentTool] = DEFAULT_TOOLS,
    max_steps: int = 10,
) -> str:
    """Run the model/tool loop until the provider returns a text response."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    messages = [ChatMessage(role="user", content=message)]
    tools_by_name: Dict[str, AgentTool] = {
        tool.definition.name: tool for tool in tools
    }
    tool_definitions = [tool.definition for tool in tools]

    for _ in range(max_steps):
        response = provider.chat(messages, tool_definitions)
        if not response.tool_calls:
            return response.content or ""

        messages.append(
            ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

        for tool_call in response.tool_calls:
            result = _execute_tool(tool_call.name, tool_call.arguments, tools_by_name)
            messages.append(
                ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                )
            )

    raise RuntimeError(f"Agent exceeded the maximum of {max_steps} model steps")


def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    tools_by_name: Dict[str, AgentTool],
) -> str:
    tool = tools_by_name.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"

    try:
        return str(tool.function(**arguments))
    except Exception as error:
        return f"Error executing tool '{name}': {error}"
