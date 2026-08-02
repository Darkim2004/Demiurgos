import ollama
import os

def run_agent(message: str) -> str:
    response = ollama.chat(
        model = os.getenv("MODEL", "qwen3:8b"),
        messages=[
            {"role": "user",
            "content": message}
        ]
    )

    # TODO: Handle errors and exceptions
    return response.message.content or ""

def main():
    print("Hello from demiurgos!")
    message = input("Enter a message: ")

    response = run_agent(message)

    print("Agent response:", response)



if __name__ == "__main__":
    main()
