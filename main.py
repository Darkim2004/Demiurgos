"""Demiurgos command-line entry point."""

from dotenv import load_dotenv

from agent import run_agent
from models.factory import create_provider


def main() -> None:
    """Read one message, run the configured agent, and print its response."""

    load_dotenv()
    print("Hello from demiurgos!")
    message = input("Enter a message: ")

    provider = create_provider()
    response = run_agent(message, provider)

    print("Agent response:", response)


if __name__ == "__main__":
    main()
