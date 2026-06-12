"""Loads per-server configs from servers/<name>.yaml.

Each YAML file describes a complete server evaluation setup: how to connect,
which models to test, eval thresholds, and inline test cases. String values
support ${ENV_VAR} interpolation for secrets.

Transport types:
- "in_process": a FastMCP server object importable in this codebase
- "http":       a remote MCP server reachable over streamable HTTP
- "stdio":      a local MCP server started via a command + args
"""

import importlib
import os
import re
from pathlib import Path

import yaml

SERVERS_DIR = Path(__file__).resolve().parent.parent / "servers"

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env(value):
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    return value


def _build_tool_source(config: dict):
    transport = config["transport"]

    if transport == "in_process":
        module = importlib.import_module(config["module"])
        return getattr(module, config["attr"])

    if transport == "http":
        from fastmcp.client.transports import StreamableHttpTransport

        return StreamableHttpTransport(
            url=config["endpoint"],
            headers=config.get("headers", {}),
        )

    if transport == "stdio":
        return {"command": config["command"], "args": config.get("args", [])}

    raise ValueError(f"Unknown transport type: {transport!r}")


def load_raw_config(name: str) -> dict:
    config_path = SERVERS_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No server config found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return _interpolate_env(config)


def load_tool_source(name: str):
    """Return just the tool source for a server (useful for one-off scripts
    like the test case generator)."""
    return _build_tool_source(load_raw_config(name))


def load_server_config(name: str) -> dict:
    """Return a structured config dict for the named server."""
    config = load_raw_config(name)

    return {
        "name": config.get("name", name),
        "tool_source": _build_tool_source(config),
        "test_cases": config.get("test_cases", []),
        "models": config.get("models", []),
        "threshold": config.get("threshold", 0.7),
        "repeats": config.get("repeats", 3),
        "metrics": config.get("metrics", ["tool_correctness"]),
        "repo": config.get("repo"),
        "setup": config.get("setup"),
        "start": config.get("start"),
        "endpoint": config.get("endpoint"),
        "coverage": config.get("coverage"),
    }


def list_servers() -> list[str]:
    """Return names of all registered servers."""
    if not SERVERS_DIR.exists():
        return []
    return sorted(p.stem for p in SERVERS_DIR.glob("*.yaml"))
