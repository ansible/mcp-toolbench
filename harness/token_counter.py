"""Count the token cost of each MCP tool definition per LLM provider.

Connects to an MCP server, discovers tools, and measures how many input
tokens each tool definition consumes in the model's context window.
"""

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from harness.mcp_client import get_tools
from harness.providers import parse_model_string

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BASELINE_MESSAGES = [{"role": "user", "content": "hi"}]


@dataclass
class ToolTokenCount:
    name: str
    tokens: int
    properties: int
    pct: float = 0.0


@dataclass
class ToolsetTokenReport:
    name: str
    tool_count: int
    total_tokens: int
    tools: list[ToolTokenCount]


@dataclass
class TokenReport:
    model: str
    server: str
    baseline_tokens: int
    tool_use_framing_tokens: int
    total_tokens: int
    total_tools: int
    timestamp: str
    toolsets: list[ToolsetTokenReport]


def _count_properties(schema: dict) -> int:
    return len(schema.get("properties", {}))


def _make_anthropic_client():
    import anthropic

    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    region = os.environ.get("CLOUD_ML_REGION", "us-east5")

    if project_id:
        return anthropic.AnthropicVertex(project_id=project_id, region=region)
    return anthropic.Anthropic()


def _get_anthropic_baseline(client, model_name: str) -> tuple[int, bool]:
    import anthropic

    try:
        resp = client.messages.count_tokens(
            model=model_name,
            messages=BASELINE_MESSAGES,
        )
        return resp.input_tokens, True
    except anthropic.BadRequestError:
        print("count_tokens API unavailable, falling back to dry-run method...")
        resp = client.messages.create(
            model=model_name,
            max_tokens=1,
            messages=BASELINE_MESSAGES,
        )
        return resp.usage.input_tokens, False


def _count_anthropic_tool(client, model_name: str, tool: dict, baseline: int, use_count_tokens: bool) -> int:
    anthropic_tool = {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": tool["parameters"],
    }
    if use_count_tokens:
        resp = client.messages.count_tokens(
            model=model_name,
            messages=BASELINE_MESSAGES,
            tools=[anthropic_tool],
        )
        return resp.input_tokens - baseline
    else:
        resp = client.messages.create(
            model=model_name,
            max_tokens=1,
            messages=BASELINE_MESSAGES,
            tools=[anthropic_tool],
        )
        return resp.usage.input_tokens - baseline


def _measure_anthropic_framing(client, model_name: str, baseline: int, use_count_tokens: bool) -> int:
    minimal_a = {"name": "_probe_a", "description": "a", "input_schema": {"type": "object", "properties": {}}}
    minimal_b = {"name": "_probe_b", "description": "b", "input_schema": {"type": "object", "properties": {}}}

    if use_count_tokens:
        r1 = client.messages.count_tokens(model=model_name, messages=BASELINE_MESSAGES, tools=[minimal_a])
        r2 = client.messages.count_tokens(model=model_name, messages=BASELINE_MESSAGES, tools=[minimal_a, minimal_b])
        one_tool = r1.input_tokens
        two_tools = r2.input_tokens
    else:
        r1 = client.messages.create(model=model_name, max_tokens=1, messages=BASELINE_MESSAGES, tools=[minimal_a])
        r2 = client.messages.create(model=model_name, max_tokens=1, messages=BASELINE_MESSAGES, tools=[minimal_a, minimal_b])
        one_tool = r1.usage.input_tokens
        two_tools = r2.usage.input_tokens

    marginal_second = two_tools - one_tool
    framing = (one_tool - baseline) - marginal_second
    return framing


def _count_anthropic(model_name: str, tools: list[dict]) -> tuple[int, int, list[ToolTokenCount]]:
    client = _make_anthropic_client()
    baseline, use_count_tokens = _get_anthropic_baseline(client, model_name)
    framing = _measure_anthropic_framing(client, model_name, baseline, use_count_tokens)

    results = []
    for tool in tools:
        raw_marginal = _count_anthropic_tool(client, model_name, tool, baseline, use_count_tokens)
        results.append(ToolTokenCount(
            name=tool["name"],
            tokens=raw_marginal - framing,
            properties=_count_properties(tool["parameters"]),
        ))

    return baseline, framing, results


def _count_openai(model_name: str, tools: list[dict]) -> tuple[int, int, list[ToolTokenCount]]:
    from openai import OpenAI

    client = OpenAI()

    baseline_resp = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, max_tokens=1)
    baseline = baseline_resp.usage.prompt_tokens

    minimal_a = {"type": "function", "function": {"name": "_probe_a", "description": "a", "parameters": {"type": "object", "properties": {}}}}
    minimal_b = {"type": "function", "function": {"name": "_probe_b", "description": "b", "parameters": {"type": "object", "properties": {}}}}
    r1 = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, tools=[minimal_a], max_tokens=1)
    r2 = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, tools=[minimal_a, minimal_b], max_tokens=1)
    framing = (r1.usage.prompt_tokens - baseline) - (r2.usage.prompt_tokens - r1.usage.prompt_tokens)

    results = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        resp = client.chat.completions.create(
            model=model_name,
            messages=BASELINE_MESSAGES,
            tools=[openai_tool],
            max_tokens=1,
        )
        raw_marginal = resp.usage.prompt_tokens - baseline
        results.append(ToolTokenCount(
            name=tool["name"],
            tokens=raw_marginal - framing,
            properties=_count_properties(tool["parameters"]),
        ))

    return baseline, framing, results


def _count_ollama(model_name: str, _tools: list[dict]) -> tuple[int, int, list[ToolTokenCount]]:
    raise NotImplementedError(
        f"Token counting is not supported for Ollama models ({model_name}). "
        "Use an Anthropic or OpenAI model instead."
    )


_COUNTERS = {
    "anthropic": _count_anthropic,
    "openai": _count_openai,
    "ollama": _count_ollama,
}


def _build_toolset_source(base_endpoint: str, toolset_name: str, raw_config: dict):
    from fastmcp.client.transports import StreamableHttpTransport

    url = f"{base_endpoint.rstrip('/')}/{toolset_name}"
    kwargs = {"url": url, "headers": raw_config.get("headers", {})}
    if raw_config.get("ssl_verify") is False:
        kwargs["verify"] = False
    return StreamableHttpTransport(**kwargs)


def count_tokens_by_toolset(
    config: dict,
    model: str,
    toolset_names: list[str],
    raw_config: dict,
) -> TokenReport:
    provider, model_name = parse_model_string(model)
    if provider not in _COUNTERS:
        raise ValueError(f"Unknown provider: {provider!r}. Use: {', '.join(_COUNTERS)}")

    base_endpoint = raw_config.get("endpoint", config.get("endpoint", ""))

    if provider == "anthropic":
        client = _make_anthropic_client()
        baseline, use_count_tokens = _get_anthropic_baseline(client, model_name)
        framing = _measure_anthropic_framing(client, model_name, baseline, use_count_tokens)

        def count_tool(tool):
            raw = _count_anthropic_tool(client, model_name, tool, baseline, use_count_tokens)
            return raw - framing

    else:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, max_tokens=1)
        baseline = resp.usage.prompt_tokens
        minimal_a = {"type": "function", "function": {"name": "_probe_a", "description": "a", "parameters": {"type": "object", "properties": {}}}}
        minimal_b = {"type": "function", "function": {"name": "_probe_b", "description": "b", "parameters": {"type": "object", "properties": {}}}}
        r1 = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, tools=[minimal_a], max_tokens=1)
        r2 = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, tools=[minimal_a, minimal_b], max_tokens=1)
        framing = (r1.usage.prompt_tokens - baseline) - (r2.usage.prompt_tokens - r1.usage.prompt_tokens)

        def count_tool(tool):
            openai_tool = {
                "type": "function",
                "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]},
            }
            r = client.chat.completions.create(model=model_name, messages=BASELINE_MESSAGES, tools=[openai_tool], max_tokens=1)
            return r.usage.prompt_tokens - baseline - framing

    toolset_reports = []
    for ts_name in toolset_names:
        tool_source = _build_toolset_source(base_endpoint, ts_name, raw_config)
        tools = asyncio.run(get_tools(tool_source))
        print(f"  {ts_name}: {len(tools)} tools")

        tool_counts = []
        for i, tool in enumerate(tools, 1):
            tokens = count_tool(tool)
            print(f"    [{i}/{len(tools)}] {tool['name']}: {tokens} tokens")
            tool_counts.append(ToolTokenCount(
                name=tool["name"],
                tokens=tokens,
                properties=_count_properties(tool["parameters"]),
            ))

        ts_total = sum(tc.tokens for tc in tool_counts)
        for tc in tool_counts:
            tc.pct = round((tc.tokens / ts_total * 100) if ts_total else 0, 1)
        tool_counts.sort(key=lambda tc: tc.tokens, reverse=True)

        toolset_reports.append(ToolsetTokenReport(
            name=ts_name,
            tool_count=len(tools),
            total_tokens=ts_total,
            tools=tool_counts,
        ))

        partial = TokenReport(
            model=model, server=config.get("name", ""),
            baseline_tokens=baseline, tool_use_framing_tokens=framing,
            total_tokens=sum(ts.total_tokens for ts in toolset_reports),
            total_tools=sum(ts.tool_count for ts in toolset_reports),
            timestamp=datetime.now(timezone.utc).isoformat(),
            toolsets=list(toolset_reports),
        )
        partial_path = RESULTS_DIR / "token-count_partial.json"
        RESULTS_DIR.mkdir(exist_ok=True)
        partial_path.write_text(json.dumps(to_json(partial), indent=2))
        print(f"  -> partial results saved to {partial_path}")

    grand_total = sum(ts.total_tokens for ts in toolset_reports)
    total_tools = sum(ts.tool_count for ts in toolset_reports)

    return TokenReport(
        model=model,
        server=config.get("name", ""),
        baseline_tokens=baseline,
        tool_use_framing_tokens=framing,
        total_tokens=grand_total,
        total_tools=total_tools,
        timestamp=datetime.now(timezone.utc).isoformat(),
        toolsets=toolset_reports,
    )


def count_tokens_single(
    tool_source,
    model: str,
    server_name: str = "",
    toolset: str | None = None,
) -> TokenReport:
    provider, model_name = parse_model_string(model)
    counter = _COUNTERS.get(provider)
    if not counter:
        raise ValueError(f"Unknown provider: {provider!r}. Use: {', '.join(_COUNTERS)}")

    tools = asyncio.run(get_tools(tool_source))
    print(f"Counting tokens for {len(tools)} tools using {model}...")

    baseline, framing, tool_counts = counter(model_name, tools)

    total = sum(tc.tokens for tc in tool_counts)
    for tc in tool_counts:
        tc.pct = round((tc.tokens / total * 100) if total else 0, 1)
    tool_counts.sort(key=lambda tc: tc.tokens, reverse=True)

    ts_name = toolset or "all"
    toolset_report = ToolsetTokenReport(
        name=ts_name,
        tool_count=len(tools),
        total_tokens=total,
        tools=tool_counts,
    )

    return TokenReport(
        model=model,
        server=server_name,
        baseline_tokens=baseline,
        tool_use_framing_tokens=framing,
        total_tokens=total,
        total_tools=len(tools),
        timestamp=datetime.now(timezone.utc).isoformat(),
        toolsets=[toolset_report],
    )


def print_table(report: TokenReport) -> None:
    print(f"\nServer: {report.server}  |  Model: {report.model}")
    print(f"Baseline (no tools): {report.baseline_tokens:,} tokens")
    print(f"Tool-use framing: {report.tool_use_framing_tokens:,} tokens (one-time cost when tools are enabled)\n")

    for ts in report.toolsets:
        print(f"  [{ts.name}] — {ts.tool_count} tools, {ts.total_tokens:,} tokens")
        print(f"  {'TOOL':<45} {'TOKENS':>6}  {'PROPS':>5}  {'%':>6}")
        for tc in ts.tools:
            print(f"    {tc.name:<43} {tc.tokens:>6}  {tc.properties:>5}  {tc.pct:>5.1f}%")
        print()

    effective = report.tool_use_framing_tokens + report.total_tokens
    print(f"Grand total: {report.total_tools} tools, {report.total_tokens:,} tokens")
    print(f"Effective total (framing + tools): {effective:,} tokens")


def render_markdown(report: TokenReport) -> str:
    effective = report.tool_use_framing_tokens + report.total_tokens
    lines = [
        f"# Token Count Report: {report.server}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Model | `{report.model}` |",
        f"| Server | {report.server} |",
        f"| Total tools | {report.total_tools} |",
        f"| Tool definitions | {report.total_tokens:,} tokens |",
        f"| Tool-use framing | {report.tool_use_framing_tokens:,} tokens |",
        f"| Effective total | {effective:,} tokens |",
        f"| Baseline (no tools) | {report.baseline_tokens:,} tokens |",
        f"| Timestamp | {report.timestamp} |",
        "",
        "> **Note:** Tool-use framing is a one-time cost the model adds when tools are",
        "> enabled. Each tool's token count below is the marginal cost of adding that",
        "> tool — the actual context consumed is framing + sum of tool definitions.",
        "",
    ]

    for ts in report.toolsets:
        lines.extend([
            f"## {ts.name}",
            "",
            f"**{ts.tool_count} tools — {ts.total_tokens:,} tokens**",
            "",
            "| Tool | Tokens | Properties | % of Toolset |",
            "|------|--------|------------|--------------|",
        ])
        for tc in ts.tools:
            lines.append(f"| `{tc.name}` | {tc.tokens:,} | {tc.properties} | {tc.pct:.1f}% |")
        lines.append("")

    # Summary table
    lines.extend([
        "## Summary",
        "",
        "| Toolset | Tools | Tokens | % of Total |",
        "|---------|-------|--------|------------|",
    ])
    for ts in sorted(report.toolsets, key=lambda t: t.total_tokens, reverse=True):
        pct = round((ts.total_tokens / report.total_tokens * 100) if report.total_tokens else 0, 1)
        lines.append(f"| {ts.name} | {ts.tool_count} | {ts.total_tokens:,} | {pct:.1f}% |")
    lines.extend([
        f"| **Tool definitions** | **{report.total_tools}** | **{report.total_tokens:,}** | |",
        f"| **+ Tool-use framing** | | **{report.tool_use_framing_tokens:,}** | *(one-time)* |",
        f"| **Effective total** | | **{effective:,}** | |",
        "",
    ])

    return "\n".join(lines)


def to_json(report: TokenReport) -> dict:
    toolsets = {}
    for ts in report.toolsets:
        toolsets[ts.name] = {
            "tool_count": ts.tool_count,
            "total_tokens": ts.total_tokens,
            "tools": [asdict(tc) for tc in ts.tools],
        }
    return {
        "model": report.model,
        "server": report.server,
        "baseline_tokens": report.baseline_tokens,
        "tool_use_framing_tokens": report.tool_use_framing_tokens,
        "total_tokens": report.total_tokens,
        "effective_total_tokens": report.tool_use_framing_tokens + report.total_tokens,
        "total_tools": report.total_tools,
        "timestamp": report.timestamp,
        "toolsets": toolsets,
    }


def save_results(report: TokenReport, fmt: str = "json") -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_model = report.model.replace(":", "-")
    timestamp = report.timestamp.replace(":", "-")
    base = f"token-count_{report.server}_{safe_model}_{timestamp}"

    if fmt == "md":
        path = RESULTS_DIR / f"{base}.md"
        path.write_text(render_markdown(report))
    else:
        path = RESULTS_DIR / f"{base}.json"
        path.write_text(json.dumps(to_json(report), indent=2))
    return path
