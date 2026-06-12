"""Runs test cases against an Ollama model and scores tool-calling accuracy."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from ollama import chat

from harness.mcp_client import get_tools

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def to_ollama_tool(tool: dict) -> dict:
    """Wrap a normalized {name, description, parameters} tool into the
    OpenAI-style format ollama.chat(tools=...) expects."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def run(tool_source, test_cases: list[dict], model: str, repeats: int = 3) -> dict:
    tools = asyncio.run(get_tools(tool_source))
    ollama_tools = [to_ollama_tool(t) for t in tools]

    results = []
    for case in test_cases:
        acceptable = {case["expected_tool"], *case.get("acceptable_tools", [])}
        runs = []
        for _ in range(repeats):
            messages = [{"role": "user", "content": case["query"]}]
            response = chat(model=model, messages=messages, tools=ollama_tools)
            chosen = [tc.function.name for tc in (response.message.tool_calls or [])]
            runs.append(chosen)

        pass_count = sum(1 for chosen in runs if any(name in acceptable for name in chosen))

        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected_tool": case["expected_tool"],
            "category": case["category"],
            "runs": runs,
            "pass_count": pass_count,
            "pass_rate": pass_count / repeats,
        })

    total_passes = sum(r["pass_count"] for r in results)
    total_runs = len(results) * repeats

    return {
        "model": model,
        "repeats": repeats,
        "tool_count": len(tools),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "accuracy": total_passes / total_runs,
    }


def print_table(run_data: dict) -> None:
    repeats = run_data["repeats"]
    print(f"{'ID':<14}{'PASS':<8}{'CATEGORY':<20}QUERY")
    for r in run_data["results"]:
        pass_str = f"{r['pass_count']}/{repeats}"
        print(f"{r['id']:<14}{pass_str:<8}{r['category']:<20}{r['query']}")

    total_passes = sum(r["pass_count"] for r in run_data["results"])
    total_runs = len(run_data["results"]) * repeats
    print(f"\nTool count: {run_data['tool_count']}")
    print(f"Overall accuracy: {total_passes}/{total_runs} ({run_data['accuracy']:.1%})")


def save_results(run_data: dict, server_name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_model = run_data["model"].replace(":", "-")
    timestamp = run_data["timestamp"].replace(":", "-")
    path = RESULTS_DIR / f"{server_name}_{safe_model}_{timestamp}.json"
    path.write_text(json.dumps(run_data, indent=2))
    return path
