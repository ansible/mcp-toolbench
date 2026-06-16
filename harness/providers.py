"""Multi-model provider abstraction for tool-calling evaluation.

Supports Anthropic (Claude), OpenAI, and Ollama. Model strings use the
format "provider:model_name" (e.g. "anthropic:claude-sonnet-4-6",
"openai:gpt-4o", "ollama:llama3.2:3b"). A plain model name without a
provider prefix defaults to Ollama for backward compatibility.
"""

import os
from dataclasses import dataclass


@dataclass
class ToolCallResult:
    name: str
    input_parameters: dict


def parse_model_string(model: str) -> tuple[str, str]:
    """Split 'provider:model_name' into (provider, model_name).

    If no provider prefix, defaults to 'ollama'.
    Handles Ollama model names that contain colons (e.g. 'llama3.2:3b')
    by only splitting on known provider prefixes.
    """
    for prefix in ("anthropic:", "openai:"):
        if model.startswith(prefix):
            return prefix[:-1], model[len(prefix):]
    if model.startswith("ollama:"):
        return "ollama", model[len("ollama:"):]
    return "ollama", model


def detect_available_providers() -> dict[str, str]:
    """Check which providers have credentials configured.

    Returns a dict of provider -> default model name for each available provider.
    """
    available = {}

    if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
        available["anthropic"] = "claude-sonnet-4-6"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        available["anthropic"] = "claude-sonnet-4-6"

    if os.environ.get("OPENAI_API_KEY"):
        available["openai"] = "gpt-4o"

    try:
        from ollama import list as ollama_list
        models = ollama_list()
        if models.get("models"):
            available["ollama"] = models["models"][0]["name"]
    except Exception:
        pass

    return available


def detect_default_model() -> str:
    """Pick the best available model based on configured credentials.

    Priority: anthropic > openai > ollama.
    Returns a provider:model string.
    """
    available = detect_available_providers()

    for provider in ("anthropic", "openai", "ollama"):
        if provider in available:
            return f"{provider}:{available[provider]}"

    return "ollama:llama3.2:3b"


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def _call_ollama(model_name: str, query: str, tools: list[dict]) -> list[ToolCallResult]:
    from ollama import chat

    ollama_tools = _to_openai_tools(tools)
    messages = [{"role": "user", "content": query}]
    response = chat(model=model_name, messages=messages, tools=ollama_tools)

    return [
        ToolCallResult(
            name=tc.function.name,
            input_parameters=tc.function.arguments if hasattr(tc.function, "arguments") else {},
        )
        for tc in (response.message.tool_calls or [])
    ]


def _call_anthropic(model_name: str, query: str, tools: list[dict]) -> list[ToolCallResult]:
    import anthropic

    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    region = os.environ.get("CLOUD_ML_REGION", "us-east5")

    if project_id:
        client = anthropic.AnthropicVertex(project_id=project_id, region=region)
    else:
        client = anthropic.Anthropic()

    anthropic_tools = _to_anthropic_tools(tools)
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        tools=anthropic_tools,
        messages=[{"role": "user", "content": query}],
    )

    return [
        ToolCallResult(
            name=block.name,
            input_parameters=block.input or {},
        )
        for block in response.content
        if block.type == "tool_use"
    ]


def _call_openai(model_name: str, query: str, tools: list[dict]) -> list[ToolCallResult]:
    import json

    from openai import OpenAI

    client = OpenAI()
    openai_tools = _to_openai_tools(tools)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        tools=openai_tools,
    )

    results = []
    for choice in response.choices:
        for tc in (choice.message.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            results.append(ToolCallResult(name=tc.function.name, input_parameters=args))
    return results


_PROVIDERS = {
    "ollama": _call_ollama,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}


def call_model(model: str, query: str, tools: list[dict]) -> list[ToolCallResult]:
    """Send a query to the specified model with tools and return tool calls.

    Args:
        model: Provider-prefixed model string (e.g. "anthropic:claude-sonnet-4-6")
               or plain Ollama model name (e.g. "llama3.2:3b").
        query: The user query to send.
        tools: List of normalized tool dicts from get_tools().

    Returns:
        List of ToolCallResult with name and input_parameters.
    """
    provider, model_name = parse_model_string(model)
    call_fn = _PROVIDERS.get(provider)
    if not call_fn:
        raise ValueError(f"Unknown provider: {provider!r}. Use: {', '.join(_PROVIDERS)}")
    return call_fn(model_name, query, tools)
