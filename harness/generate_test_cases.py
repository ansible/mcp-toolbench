"""Experiment: auto-generate (query, expected_tool) test cases for a subset
of an external MCP server's tools, using an LLM to write the queries.

For each tool, asks the model for two queries that should both trigger that
tool:
  - "easy": closely related to the tool's name/description
  - "hard": no obvious keyword overlap, but should still imply this tool

Generated cases are tagged source="generated" so they can be reviewed/pruned
before being trusted as ground truth - an LLM-written query could turn out
to be genuinely ambiguous with a similar tool.
"""

import asyncio
import json

from ollama import chat

from harness.config import CONFIGS_DIR, load_tool_source
from harness.mcp_client import get_tools

# Small batch: includes some near-duplicate / confusable tools on purpose
TARGET_TOOLS = [
    "job_templates_list",
    "workflow_job_templates_list",
    "jobs_list",
    "inventories_list",
    "hosts_list",
    "credentials_list",
    "users_list",
    "activation_instances_list",
]

GEN_MODEL = "qwen3.5:4b"

PROMPT_TEMPLATE = """You are creating test cases for a tool-calling accuracy benchmark.

Tool name: {name}
Tool description: {description}

Generate two example user queries (natural, conversational, as if typed by a
real user) that should both result in this exact tool being called - and no
other tool.

1. "easy": a direct query closely related to the tool's name/description.
2. "hard": a query that does NOT share obvious keywords with the tool's name
   or description, but a person asking it would still expect this tool to be
   used (e.g. a synonym, indirect phrasing, or a different framing of the
   same need).

Respond with JSON only: {{"easy": "...", "hard": "..."}}"""

RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "easy": {"type": "string"},
        "hard": {"type": "string"},
    },
    "required": ["easy", "hard"],
}


def generate_for_tool(tool: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(name=tool["name"], description=tool["description"])
    response = chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=RESPONSE_FORMAT,
        think=False,
    )
    return json.loads(response.message.content)


def main():
    tool_source = load_tool_source("aap")
    all_tools = asyncio.run(get_tools(tool_source))
    by_name = {t["name"]: t for t in all_tools}

    cases = []
    for i, name in enumerate(TARGET_TOOLS, start=1):
        tool = by_name[name]
        generated = generate_for_tool(tool)

        for variant in ("easy", "hard"):
            cases.append({
                "id": f"AAP{i:02d}-{variant}",
                "query": generated[variant],
                "expected_tool": name,
                "acceptable_tools": [],
                "category": variant,
                "source": "generated",
            })

    out_path = CONFIGS_DIR / "aap_test_cases.json"
    out_path.write_text(json.dumps(cases, indent=2))

    for c in cases:
        print(f"{c['id']:<14} [{c['expected_tool']}]")
        print(f"   {c['query']}")

    print(f"\nWrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
