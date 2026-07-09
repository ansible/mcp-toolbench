# How Token Counting Works

## The Problem

When an MCP client connects to a server and discovers tools, those tool definitions
(name, description, input schema) get injected into the LLM's context window. Every
tool consumes tokens just by being available — before a user even asks a question.
With 80+ tools across multiple toolsets, this overhead is significant and invisible
without measurement.

The `token-count` subcommand measures the exact per-tool token cost so you can make
informed decisions about toolset composition, schema complexity, and context budgets.

## How It Works

### What the LLM Actually Sees

Tool definitions aren't passed to the model as raw JSON. The provider serializes them
into an internal text format before tokenization. For example, a tool like:

```json
{
  "name": "hosts_list",
  "description": "List all hosts.",
  "input_schema": {
    "type": "object",
    "properties": {
      "page": {"type": "number"},
      "search": {"type": "string"}
    }
  }
}
```

gets expanded into a provider-specific representation (XML-like tags for Anthropic,
structured text for OpenAI) that is typically longer than the JSON. The exact format
is internal and undocumented, which is why we measure tokens through the API rather
than estimating from byte length.

### The Three Components of Tool Token Cost

A request with tools enabled has three token components:

```
total input tokens = baseline + tool-use framing + tool definitions
```

| Component | What it is | Behavior |
|-----------|-----------|----------|
| **Baseline** | The user message and any system prompt | Fixed per request, independent of tools |
| **Tool-use framing** | System instructions the model injects to explain how tool calling works | One-time fixed cost (~497 tokens on Claude Sonnet 4.6). Present whenever `tools` is non-empty, regardless of how many tools |
| **Tool definitions** | The serialized name + description + schema of each tool | Additive — each tool contributes its own marginal cost |

### Measuring the Framing Overhead

The framing cost can't be read directly from the API. We isolate it with two
probe requests using symmetric minimal tools:

```
request with 0 tools:  8 tokens   (baseline)
request with 1 probe: 541 tokens  (baseline + framing + probe_content)
request with 2 probes: 577 tokens (baseline + framing + probe_content × 2)
```

The probes are identical in structure (`name: _probe_a/_probe_b`, `description: a/b`,
empty schema), so their token content is effectively equal. This lets us solve:

```
second_probe_marginal = 577 - 541 = 36 tokens
first_probe_raw       = 541 - 8   = 533 tokens
framing               = 533 - 36  = 497 tokens
```

The framing is what the first tool "pays" that the second tool doesn't — the
model's internal instructions about how to use tools.

### Measuring Per-Tool Cost

With the framing isolated, each tool's true marginal cost is:

```
tool_marginal = (request with 1 tool) - baseline - framing
```

For example, `config_retrieve` with an empty schema:

```
request with config_retrieve: 543 tokens
baseline:                       8 tokens
framing:                      497 tokens
marginal:                      38 tokens
```

### Verification

This methodology has been verified to be exactly additive. For any set of N tools:

```
framing + sum(tool_marginals) = actual_combined_request - baseline
```

Tested with 3 tools of varying complexity (0, 1, and 7 schema properties):

```
config_retrieve:  38 tokens (0 properties)
status_retrieve:  53 tokens (1 property)
jobs_list:       108 tokens (7 properties)

Predicted: 497 + 38 + 53 + 108 = 696
Actual:    704 - 8              = 696
Error:     0
```

The per-tool marginals also match exactly when verified by incrementally adding
tools to a combined request (where the 2nd+ tools have no framing component).

### Provider Differences

**Anthropic (Claude):** Uses the `count_tokens` API when available (exact, no cost).
Falls back to a dry-run method (`max_tokens=1` real request) when `count_tokens` is
blocked (e.g., Vertex AI org policies). Both methods return identical `input_tokens`
values since input token count is independent of output length.

**OpenAI:** No public `count_tokens` endpoint for tools. Uses the dry-run method
(`max_tokens=1`) and reads `usage.prompt_tokens`. Costs one output token per
measurement but gives exact counts.

**Ollama:** Not supported. Local models don't expose token counting APIs, and
running a local tokenizer would require replicating the model's internal tool
serialization format.

### What This Measures vs. What It Doesn't

**Measured (deterministic, pre-computable):**
- Token cost of tool definitions in the context window
- Tool-use framing overhead
- Baseline message cost

**Not measured (variable at runtime):**
- LLM output tokens when choosing/calling a tool (non-deterministic)
- Tool execution result size (depends on the API response)
- Follow-up conversation tokens after tool results are returned

## Usage

```bash
# Count all toolsets for a server
python main.py token-count aap --model anthropic:claude-sonnet-4-6

# Count a specific toolset
python main.py token-count aap --model anthropic:claude-sonnet-4-6 --toolset job_management

# Output as markdown
python main.py token-count aap --model anthropic:claude-sonnet-4-6 --output md

# Save results
python main.py token-count aap --model anthropic:claude-sonnet-4-6 --save-format md
```

## Reading the Results

The report groups tools by toolset and shows:

- **Tokens**: the marginal cost of including this tool (framing excluded)
- **Properties**: number of schema properties (correlates with token cost)
- **% of Toolset**: how much of the toolset's budget this tool consumes

The summary table at the bottom shows toolset-level totals and the framing
overhead as a separate line item:

```
| Toolset              | Tools | Tokens |
|----------------------|-------|--------|
| user_management      | 18    | 9,303  |
| job_management       | 23    | 3,885  |
| ...                  |       |        |
| Tool definitions     | 81    | 22,234 |
| + Tool-use framing   |       | 497    |  (one-time)
| Effective total      |       | 22,731 |
```

The **effective total** is what the model actually sees in its context window
when all toolsets are loaded: framing (once) + all tool definitions.
