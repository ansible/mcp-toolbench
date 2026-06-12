"""Connects to an MCP server and returns its tools in a normalized form:

    {"name": ..., "description": ..., "parameters": {<json schema>}}

This shape is intentionally close to the OpenAI/Ollama function-calling
format, so it can be fed almost directly into `ollama.chat(tools=...)`.
"""

import asyncio

from fastmcp import Client


async def get_tools(server) -> list[dict]:
    """Connect to `server` (anything fastmcp.Client accepts: a FastMCP
    instance for in-process use, a script path, or a URL) and return its
    tools as a list of normalized dicts.
    """
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


if __name__ == "__main__":
    import json
    from server.tools_server import mcp

    normalized = asyncio.run(get_tools(mcp))
    print(json.dumps(normalized, indent=2))
