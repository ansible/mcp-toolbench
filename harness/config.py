"""Loads per-server configs from configs/<name>.json.

Each config describes how to connect to an MCP server (its "tool source")
and which test cases file to use. Three transport types are supported:

- "in_process": a FastMCP server object importable in this codebase
                 (e.g. our dummy test fixture)
- "http":       a remote MCP server reachable over streamable HTTP, with
                 optional headers (e.g. an Authorization bearer token)
- "stdio":      a local MCP server started via a command + args
"""

import importlib
import json
from pathlib import Path

from fastmcp.client.transports import StreamableHttpTransport

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _build_tool_source(config: dict):
    transport = config["transport"]

    if transport == "in_process":
        module = importlib.import_module(config["module"])
        return getattr(module, config["attr"])

    if transport == "http":
        return StreamableHttpTransport(
            url=config["url"],
            headers=config.get("headers", {}),
        )

    if transport == "stdio":
        return {"command": config["command"], "args": config.get("args", [])}

    raise ValueError(f"Unknown transport type: {transport!r}")


def load_raw_config(name: str) -> dict:
    config_path = CONFIGS_DIR / f"{name}.json"
    return json.loads(config_path.read_text())


def load_tool_source(name: str):
    """Return just the tool source for `name` (useful for one-off scripts
    like the test case generator, which don't need test cases)."""
    return _build_tool_source(load_raw_config(name))


def load_server_config(name: str) -> dict:
    """Return {"name", "tool_source", "test_cases"} for the named server."""
    config = load_raw_config(name)
    test_cases_path = CONFIGS_DIR.parent / config["test_cases_file"]

    return {
        "name": config.get("name", name),
        "tool_source": _build_tool_source(config),
        "test_cases": json.loads(test_cases_path.read_text()),
    }
