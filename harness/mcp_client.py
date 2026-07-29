"""Connects to an MCP server and returns its tools in a normalized form:

    {"name": ..., "description": ..., "parameters": {<json schema>}}

This shape is intentionally close to the OpenAI/Ollama function-calling
format, so it can be fed almost directly into `ollama.chat(tools=...)`.
"""

import asyncio
from contextlib import asynccontextmanager


async def get_tools(server) -> list[dict]:
    """Connect to `server` (anything fastmcp.Client accepts: a FastMCP
    instance for in-process use, a script path, or a URL) and return its
    tools as a list of normalized dicts.
    """
    from fastmcp import Client

    async with Client(server) as client:
        tools = await client.list_tools()

    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in tools
    ]


@asynccontextmanager
async def mcp_session(server):
    """Keep an MCP client alive for multi-turn tool calling."""
    from fastmcp import Client

    async with Client(server) as client:
        yield client


def normalize_tools(tools) -> list[dict]:
    """Convert fastmcp tool objects to normalized dicts."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in tools
    ]


async def execute_tool(client, name: str, arguments: dict) -> str:
    """Call a tool on the MCP server and return the result as text."""
    result = await client.call_tool(name, arguments)
    content = result if isinstance(result, list) else getattr(result, "content", [result])
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


if __name__ == "__main__":
    import json
    from server.tools_server import mcp

    normalized = asyncio.run(get_tools(mcp))
    print(json.dumps(normalized, indent=2))
