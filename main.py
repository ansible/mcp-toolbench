import argparse
import json
import sys

from harness.config import list_servers, load_server_config
from harness.lifecycle import setup_server, teardown_server
from harness.runner import print_table, render_markdown, run, save_results


def cmd_run(args):
    from harness.providers import detect_default_model

    config = load_server_config(args.server)
    models = [args.model] if args.model else config["models"]
    if not models:
        default = detect_default_model()
        print(f"No model specified — auto-detected: {default}")
        models = [default]

    repeats = args.repeats or config["repeats"]
    threshold = args.threshold or config["threshold"]
    metric_names = [m.strip() for m in args.metrics.split(",")] if args.metrics else config["metrics"]

    process = setup_server(config)
    failed = False

    try:
        for model in models:
            if args.output != "json":
                print(f"\n{'='*60}")
                print(f"Server: {config['name']}  |  Model: {model}")
                print(f"{'='*60}\n")

            run_data = run(
                config["tool_source"],
                config["test_cases"],
                model,
                repeats=repeats,
                metric_names=metric_names,
                threshold=threshold,
                judge_model=args.judge_model,
            )

            if args.output == "json":
                print(json.dumps(run_data, indent=2))
            elif args.output == "md":
                print(render_markdown(run_data))
            else:
                print_table(run_data)

            if args.output != "json":
                save_fmt = args.save_format or ("md" if args.output == "md" else "json")
                path = save_results(run_data, config["name"], fmt=save_fmt)
                print(f"\nSaved results to {path}")

            if args.ci and run_data["accuracy"] < threshold:
                failed = True
    finally:
        teardown_server(process, config)

    if failed:
        print(f"\nCI FAILED: one or more models below threshold {threshold:.1%}")
        sys.exit(1)


def cmd_run_all(args):
    servers = list_servers()
    if not servers:
        print("No servers registered in servers/", file=sys.stderr)
        sys.exit(1)

    print(f"Running evals for {len(servers)} servers: {', '.join(servers)}\n")

    failed = []
    for server_name in servers:
        args.server = server_name
        try:
            cmd_run(args)
        except SystemExit as e:
            if e.code == 1:
                failed.append(server_name)
        except Exception as e:
            print(f"\nError running {server_name}: {e}", file=sys.stderr)
            failed.append(server_name)

    if failed:
        print(f"\nFailed servers: {', '.join(failed)}")
        sys.exit(1)


def cmd_list(args):
    from harness.config import load_raw_config
    from harness.providers import detect_available_providers, detect_default_model

    servers = list_servers()
    if not servers:
        print("No servers registered in servers/")
        return

    available = detect_available_providers()
    if available:
        print(f"Detected providers: {', '.join(available.keys())}")
        print(f"Default model: {detect_default_model()}\n")
    else:
        print("No cloud providers detected — defaulting to local Ollama\n")

    print(f"{'NAME':<20}{'TRANSPORT':<15}{'MODELS':<40}{'TESTS':<8}")
    for name in servers:
        config = load_raw_config(name)
        models = config.get("models", [])
        models_str = ", ".join(models) if models else "(auto-detect)"
        test_count = len(config.get("test_cases", []))
        transport = config.get("transport", "unknown")
        print(f"{name:<20}{transport:<15}{models_str:<40}{test_count:<8}")


def _resolve_coverage(config: dict, cli_count: int | None, total_tools: int) -> int:
    """Determine how many tools to generate tests for.

    Priority: --count CLI flag > coverage YAML field > all tools.
    """
    if cli_count is not None:
        return min(cli_count, total_tools)

    coverage = config.get("coverage")
    if coverage is None:
        return total_tools
    if isinstance(coverage, str) and coverage.lower() == "all":
        return total_tools
    if isinstance(coverage, int):
        return min(coverage, total_tools)

    return total_tools


GEN_PROMPT = """You are creating test cases for a tool-calling accuracy benchmark.

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


def _generate_queries(model_str: str, tool: dict) -> dict:
    """Ask an LLM to generate easy/hard queries for a tool via text completion (no tool-calling)."""
    import json as _json
    import os

    from harness.providers import parse_model_string

    provider, model_name = parse_model_string(model_str)
    prompt = GEN_PROMPT.format(name=tool["name"], description=tool["description"])

    if provider == "anthropic":
        import anthropic

        project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        region = os.environ.get("CLOUD_ML_REGION", "us-east5")
        if project_id:
            client = anthropic.AnthropicVertex(project_id=project_id, region=region)
        else:
            client = anthropic.Anthropic()

        response = client.messages.create(
            model=model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        text = response.choices[0].message.content

    elif provider == "ollama":
        from ollama import chat

        response = chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return _json.loads(text[start:end])
    except (ValueError, _json.JSONDecodeError):
        return {"easy": f"Use {tool['name']}", "hard": f"Use {tool['name']} indirectly"}


def cmd_generate(args):
    from harness.config import SERVERS_DIR
    from harness.mcp_client import get_tools
    from harness.providers import detect_default_model

    import asyncio

    import yaml

    config = load_server_config(args.server)
    process = setup_server(config)

    try:
        tools = asyncio.run(get_tools(config["tool_source"]))
        model = args.model or (config["models"][0] if config["models"] else None)
        if not model:
            model = detect_default_model()
            print(f"No model specified — auto-detected: {model}")

        count = _resolve_coverage(config, args.count, len(tools))

        static_ids = {c["id"] for c in config.get("test_cases", [])}
        print(f"Generating test cases for {count}/{len(tools)} tools using {model}...")
        print(f"Static test cases in {args.server}.yaml: {len(static_ids)}")

        cases = []
        for i, tool in enumerate(tools[:count], start=1):
            parsed = _generate_queries(model, tool)

            for variant in ("easy", "hard"):
                case_id = f"GEN{i:02d}-{variant}"
                query = parsed.get(variant, f"Use {tool['name']}")
                cases.append({
                    "id": case_id,
                    "query": query,
                    "expected_tool": tool["name"],
                    "category": variant,
                    "source": "generated",
                })

        out_path = SERVERS_DIR / f"{args.server}.generated.yaml"
        with open(out_path, "w") as f:
            yaml.dump({"test_cases": cases}, f, default_flow_style=False, sort_keys=False)

        print(f"\nGenerated {len(cases)} test cases -> {out_path}")
        print(f"At runtime, these merge with the {len(static_ids)} static cases from {args.server}.yaml")
    finally:
        teardown_server(process, config)


def main():
    parser = argparse.ArgumentParser(description="MCP tool-calling evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared flags
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", default=None, help="Override model (provider:name format)")
    shared.add_argument("--repeats", type=int, default=None, help="Runs per test case")
    shared.add_argument("--metrics", default=None, help="Comma-separated metrics")
    shared.add_argument("--threshold", type=float, default=None, help="Pass/fail threshold 0.0-1.0")
    shared.add_argument("--judge-model", default=None, help="Model for LLM-as-judge metrics")
    shared.add_argument("--ci", action="store_true", help="Exit with code 1 if below threshold")
    shared.add_argument("--output", choices=["table", "json", "md"], default="table", help="Output format")
    shared.add_argument("--save-format", choices=["json", "md"], default=None, help="Saved file format (defaults to match --output)")

    # run <server>
    run_parser = subparsers.add_parser("run", parents=[shared], help="Run evals for a server")
    run_parser.add_argument("server", help="Server name (matches servers/<name>.yaml)")

    # run-all
    subparsers.add_parser("run-all", parents=[shared], help="Run evals for all registered servers")

    # list
    subparsers.add_parser("list", help="List registered servers")

    # generate <server>
    gen_parser = subparsers.add_parser("generate", help="Generate test cases from tool discovery")
    gen_parser.add_argument("server", help="Server name")
    gen_parser.add_argument("--model", default=None, help="Model for generation")
    gen_parser.add_argument("--count", type=int, default=None, help="Override coverage — max tools to generate for")

    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "run-all": cmd_run_all,
        "list": cmd_list,
        "generate": cmd_generate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
