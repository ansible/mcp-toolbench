# mcp-toolbench

Org-wide MCP tool-calling evaluation harness. Register any MCP server via a
YAML config, and the harness will connect to it, discover its tools, run
queries through various LLMs, and score tool-calling accuracy using
[DeepEval](https://github.com/confident-ai/deepeval) metrics.

## Stack

- **DeepEval** — scoring engine (ToolCorrectnessMetric, ArgumentCorrectnessMetric, MCPUseMetric)
- **FastMCP** — MCP client for tool discovery and server connectivity
- **Multi-model providers** — Anthropic (Vertex AI + direct API), OpenAI, Ollama
- **YAML eval registry** — each eval gets a config file with connection details, test cases, and eval settings

## Project structure

```
mcp-toolbench/
├── evals/
│   ├── aap.yaml                 # AAP MCP server config + test cases
│   └── dummy.yaml               # dummy in-process server for local testing
├── harness/
│   ├── config.py                # YAML config loader with ${ENV_VAR} interpolation
│   ├── providers.py             # multi-model provider abstraction + auto-detection
│   ├── runner.py                # DeepEval scoring engine + result formatting
│   ├── mcp_client.py            # MCP tool discovery
│   ├── lifecycle.py             # server clone/setup/start/stop lifecycle
│   └── generate_test_cases.py   # standalone test case generator
├── server/
│   └── tools_server.py          # dummy in-process FastMCP server (6 tools)
├── results/                     # JSON/Markdown results per run
├── main.py                      # CLI entrypoint
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The harness auto-detects available providers based on environment variables:

| Provider | Required env vars |
|----------|-------------------|
| Anthropic (Vertex AI) | `ANTHROPIC_VERTEX_PROJECT_ID` + `CLOUD_ML_REGION` |
| Anthropic (direct) | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Ollama | Ollama running locally (no env vars needed) |

If no cloud credentials are found, it defaults to local Ollama.

## Usage

```bash
# List registered evals and detected providers
python main.py list

# Run an eval (uses models from YAML config)
python main.py run aap

# Override model
python main.py run aap --model anthropic:claude-sonnet-4-6 --repeats 1

# Run all registered evals
python main.py run-all

# CI mode — exits non-zero if below threshold
python main.py run aap --ci --threshold 0.8

# Output as markdown
python main.py run aap --output md

# Save results as markdown file
python main.py run aap --save-format md

# Generate test cases from discovered tools
python main.py generate aap --model anthropic:claude-sonnet-4-6
```

## Eval config format (`evals/<name>.yaml`)

```yaml
name: my-eval

# Optional: repo cloning + server lifecycle
repo: https://github.com/org/my-mcp-server
setup: npm install
start: npm start

# Connection
transport: http                          # http | in_process | stdio
endpoint: http://localhost:3000/mcp
headers:
  Authorization: "Bearer ${MY_TOKEN}"    # env var interpolation

# Eval settings
models:
  - anthropic:claude-sonnet-4-6
  - openai:gpt-4o
coverage: all          # all | <number> — tools to cover when generating tests
threshold: 0.7
repeats: 3
metrics:
  - tool_correctness

# Test cases
test_cases:
  - id: TC01-easy
    query: "List my available job templates."
    expected_tool: job_templates_list
    category: easy
```

### Coverage

The `coverage` field controls how many tools `python main.py generate` creates test cases for:

- `coverage: all` — every discovered tool
- `coverage: 20` — 20 tools
- omitted — defaults to all
- `--count` CLI flag overrides the YAML value

Each tool gets an easy (keyword-matching) and hard (paraphrased) query.

### Static vs generated test cases

Test cases live in two places:

- **`evals/<name>.yaml`** — hand-curated static cases, committed to git. These are your ground truth.
- **`evals/<name>.generated.yaml`** — LLM-generated cases, created by `python main.py generate <name>`. Gitignored.

At runtime, both are merged. Static cases take priority — if a generated case has the same ID as a static one, the generated case is skipped.

### Markdown reports

Use `--output md` to print results as markdown, or `--save-format md` to save a `.md` file. Markdown reports include a **Recommendations** section that analyzes failure patterns:

- List vs retrieve tool confusion
- Frequently confused tool pairs
- Queries where no tool was selected
- Easy vs hard accuracy breakdown
- Consistently failing tools

### Transport types

- **`http`** — remote MCP server over streamable HTTP, with optional auth headers
- **`in_process`** — a FastMCP server object importable in this codebase
- **`stdio`** — a local MCP server started via command + args

## Scoring

For each test case, the model receives the full tool list from the MCP server
and a natural-language query. A run passes if the model's chosen tool(s)
include the expected tool. DeepEval metrics provide additional scoring
(tool correctness, argument correctness, MCP use).

Results are saved to `results/` as JSON or Markdown.

## Adding a new eval

1. Create `evals/<name>.yaml` with connection details and hand-written test cases
2. Run `python main.py generate <name>` to auto-generate additional cases into `<name>.generated.yaml`
3. Run `python main.py run <name>` — static + generated cases are merged automatically
