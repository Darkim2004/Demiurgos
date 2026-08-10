# Demiurgos

Demiurgos is a small provider-independent agent loop. The agent works only with
types owned by this project; Ollama and Groq are adapters at the application
boundary.

```text
main.py -> agent.py -> ModelProvider
                         |-- OllamaProvider
                         `-- GroqProvider
```

## Setup

The project requires Python 3.10 or newer and uses `uv` for dependency and
environment management.

```powershell
uv sync --dev
Copy-Item .env.example .env
```

The CLI loads `.env` automatically. Select one of the following configurations.

For Ollama:

```text
MODEL_PROVIDER="ollama"
MODEL="qwen3:8b"
```

For Groq:

```text
MODEL_PROVIDER="groq"
MODEL="openai/gpt-oss-20b"
GROQ_API_KEY="your-api-key"
```

Then run:

```powershell
uv run python main.py
```

`MODEL_PROVIDER` defaults to `ollama`, and the Ollama model defaults to
`qwen3:8b`. Groq requires both `MODEL` and `GROQ_API_KEY`.

## Architecture

- `models/base.py` defines `ModelProvider`, `ModelResponse`, `ToolCall`, and the
  provider-independent conversation/tool types.
- `models/ollama_provider.py` and `models/groq_provider.py` translate native SDK
  requests and responses without executing tools.
- `agent.py` owns the model/tool loop and has no provider SDK imports.
- `models/factory.py` is the only place that branches on `MODEL_PROVIDER`.
- `tools/` contains local tool implementations.

## Tests

The default suite is offline and uses fake clients/providers:

```powershell
uv run pytest -v
```

Live checks are opt-in. Configure the corresponding provider first, then set
one of these process variables before running the integration tests:

```powershell
# Ollama
$env:RUN_OLLAMA_INTEGRATION = "1"
uv run pytest -v tests/test_integration.py -k ollama

# Groq
$env:RUN_GROQ_INTEGRATION = "1"
uv run pytest -v tests/test_integration.py -k groq
```
