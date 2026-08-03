import ollama
import os
from typing import Any, Callable, List, Mapping, Union


ChatMessage = Union[Mapping[str, Any], ollama.Message]
ToolFunction = Callable[..., int]

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

AVAILABLE_TOOLS: Mapping[str, ToolFunction] = {
    "add": add,
}


def main() -> None:
    print("Hello from demiurgos!")
    message = input("Enter a message: ")

    messages: List[ChatMessage] = [
        {
            "role": "user",
            "content": message,
        }
    ]

    response = ollama.chat(
            model = os.getenv("MODEL", "qwen3:8b"),
            messages= messages,
            tools=[add]
    )

    messages.append(response.message)

    for i in response.message.tool_calls or ():
        tool_name = i.function.name
        tool_args = i.function.arguments
        tool_function = AVAILABLE_TOOLS.get(tool_name)

        if tool_function is None:
            print(f"Unknown tool call: {tool_name}")
        else:
            tool_result = tool_function(**tool_args)
            print(f"Tool call: {tool_name}({tool_args}) -> {tool_result}")

        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": str(tool_result),
        })

    final_response = ollama.chat(
        model=os.getenv("MODEL", "qwen3:8b"),
        messages=messages,
    )

    print("Agent response:", final_response.message.content)



if __name__ == "__main__":
    main()
