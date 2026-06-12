# tool-test

A local tool-calling accuracy harness. It connects to **any MCP server**,
sends a set of natural-language test queries to an Ollama model with that
server's tools available, and scores how often the model picks the
*expected* tool. RAG retrieval and tool execution are out of scope - this
only measures **tool selection** accuracy.

## Stack

- **Ollama** - local model inference (must already be installed and running)
- **FastMCP** - MCP client used to connect to servers and list their tools;
  also hosts a small dummy MCP server used as a test fixture
- **harness** - config-driven runner: connects to a server, normalizes its
  tool list, runs test cases against a model, and scores the results

## Project structure

```
tool-test/
├── server/
│   └── tools_server.py        # dummy in-process FastMCP server (6 tools)
├── harness/
│   ├── config.py               # loads configs/<name>.json -> tool source + test cases
│   ├── mcp_client.py            # connects to an MCP server, returns normalized tools
│   ├── runner.py                 # runs test cases, scores, saves results
│   └── generate_test_cases.py    # LLM-assisted test case generation (needs review)
├── configs/
│   ├── dummy.json                 # config for the in-process dummy server
│   ├── dummy_test_cases.json       # 15 hand-written test cases for the dummy server
│   ├── aap.json                     # config for a real MCP server (AAP, over HTTP)
│   └── aap_test_cases.json           # LLM-generated + reviewed test cases for AAP
├── results/                          # JSON results per run (created on first run)
└── main.py                            # CLI entrypoint
```

## Setup

1. Make sure Ollama is installed and running, and pull the model(s) you want
   to test, e.g.:

   ```bash
   ollama pull qwen3.5:4b
   ollama pull llama3.2:3b
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

```bash
python main.py --server dummy --model qwen3.5:4b
python main.py --server dummy --model llama3.2:3b --repeats 5
python main.py --server aap --model llama3.2:3b
```

- `--server` - name of a config file in `configs/<name>.json` (e.g. `dummy`,
  `aap`)
- `--model` - any Ollama model name you've pulled
- `--repeats` - how many times to run each test case (default 3). Model
  output is non-deterministic, so a single run can be misleading - repeats
  give a per-test pass *rate* instead of a binary pass/fail.

Each run prints a results table (pass rate per test case + overall accuracy)
and saves the full results, including every individual run's tool choice and
the total tool count exposed by the server, to
`results/<server>_<model>_<timestamp>.json` for later comparison.

## Connecting to an MCP server

Each server is described by a `configs/<name>.json` file with a transport
type and a path to its test cases file. Three transport types are supported
(see `harness/config.py`):

- **`in_process`** - a FastMCP server object importable in this codebase.
  Used for the dummy fixture:

  ```json
  {
    "name": "dummy",
    "transport": "in_process",
    "module": "server.tools_server",
    "attr": "mcp",
    "test_cases_file": "configs/dummy_test_cases.json"
  }
  ```

- **`http`** - a remote MCP server reachable over streamable HTTP, with
  optional headers (e.g. an `Authorization` bearer token). Used for the AAP
  MCP server:

  ```json
  {
    "name": "aap",
    "transport": "http",
    "url": "http://localhost:3000/mcp",
    "headers": { "Authorization": "Bearer dummy-token" },
    "test_cases_file": "configs/aap_test_cases.json"
  }
  ```

  The AAP server validates its bearer token against a real AAP instance
  (`BASE_URL`). For local testing, point `BASE_URL` at the bundled mock AAP
  server (`npm run simulate:mock-aap-server`), which accepts any token - the
  exact token value doesn't matter, only that *something* is present and the
  validation request succeeds.

- **`stdio`** - a local MCP server started via a command + args. Implemented
  in `harness/config.py` but not yet exercised against a real server.

  ```json
  {
    "name": "some-server",
    "transport": "stdio",
    "command": "node",
    "args": ["path/to/server.js"],
    "test_cases_file": "configs/some-server_test_cases.json"
  }
  ```

To add a new server: write its config JSON, write or generate a test cases
file for it, and run `python main.py --server <name> --model <model>`.

## Test case format

Each test cases file is a JSON array of:

```json
{
  "id": "TC01",
  "query": "What's the temperature in Paris right now?",
  "expected_tool": "get_current_temperature",
  "acceptable_tools": [],
  "category": "exact_match"
}
```

- `acceptable_tools` - additional tool names that also count as a pass (for
  deliberately ambiguous cases)
- `category` - free-form label used for grouping in the results table.
  Hand-written cases use `exact_match` / `no_keyword_overlap` / `ambiguous`;
  generated cases use `easy` / `hard`
- `source` - optional, set to `"generated"` by `generate_test_cases.py` to
  flag cases that need manual review; absent on hand-written cases

## Generating test cases with an LLM

`harness/generate_test_cases.py` asks a model to write an "easy" (keyword-
overlapping) and a "hard" (no-keyword-overlap, paraphrased) query for each
tool in `TARGET_TOOLS`, tags them `source: "generated"`, and writes them to
a test cases file.

**Generated cases are a starting point, not ground truth** - review them
before trusting them. In practice, generated "hard" queries sometimes
interpret a tool's generic-English words (e.g. "jobs", "inventory") in their
everyday sense rather than the server's domain-specific sense, producing
queries that wouldn't map to the expected tool in real use. A couple of such
cases were removed from `configs/aap_test_cases.json` after review.

## How scoring works

For each test case, the model is given the full list of tools from the
target MCP server and a natural-language query, run `--repeats` times. A run
"passes" if the model's chosen tool(s) include the expected tool (or one of
the listed acceptable alternatives). The per-case score is `pass_count /
repeats`, and the overall accuracy is total passes over total runs.

## Example results

- **dummy** (6 tools, 15 hand-written cases, `llama3.2:3b`): ~93% accuracy
- **aap** (62 tools, 14 generated cases, `llama3.2:3b`): ~14% accuracy

The gap is the headline finding so far: with a large tool list, the model
frequently calls *no tool at all*, or picks a confusable sibling tool (e.g.
`credentials_retrieve` instead of `credentials_list`) - exactly the kind of
issue this harness is meant to surface.

## Status / what's not here yet

- RAG-based tool retrieval (pass only the top-k relevant tools instead of
  all of them, and measure retrieval recall separately from selection
  accuracy)
- Confusable-tool-pair diagnostics based on tool description similarity
- LLM-generated query paraphrases to expand existing seed cases
- `stdio` transport is implemented but untested against a real server
