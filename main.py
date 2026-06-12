import argparse

from harness.config import load_server_config
from harness.runner import print_table, run, save_results


def main():
    parser = argparse.ArgumentParser(description="Tool-calling accuracy test harness")
    parser.add_argument("--server", required=True, help="Server config name (configs/<name>.json), e.g. dummy, aap")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. qwen3.5:4b")
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to run each test case")
    args = parser.parse_args()

    server_config = load_server_config(args.server)

    run_data = run(server_config["tool_source"], server_config["test_cases"], args.model, repeats=args.repeats)
    print_table(run_data)

    path = save_results(run_data, server_config["name"])
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    main()
