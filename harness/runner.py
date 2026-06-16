"""Runs test cases against an LLM and scores tool-calling accuracy using DeepEval."""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.mcp_client import get_tools
from harness.providers import call_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _build_expected_tools(case: dict) -> list:
    from deepeval.test_case import ToolCall

    tools = [ToolCall(name=case["expected_tool"])]
    for name in case.get("acceptable_tools", []):
        tools.append(ToolCall(name=name))
    return tools


def _resolve_judge_model(model_str: str):
    """Convert a provider:model string into a DeepEval model object."""
    import os

    from harness.providers import parse_model_string, detect_available_providers

    provider, model_name = parse_model_string(model_str)

    if provider == "anthropic":
        available = detect_available_providers()
        project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        region = os.environ.get("CLOUD_ML_REGION", "us-east5")

        if project_id:
            from deepeval.models import AnthropicModel

            class AnthropicVertexModel(AnthropicModel):
                def __init__(self, model, project_id, region):
                    self._vertex_project = project_id
                    self._vertex_region = region
                    super().__init__(model=model)

                def load_model(self, async_mode=False):
                    import anthropic
                    if async_mode:
                        return anthropic.AsyncAnthropicVertex(
                            project_id=self._vertex_project,
                            region=self._vertex_region,
                        )
                    return anthropic.AnthropicVertex(
                        project_id=self._vertex_project,
                        region=self._vertex_region,
                    )

            return AnthropicVertexModel(model_name, project_id, region)
        else:
            from deepeval.models import AnthropicModel
            return AnthropicModel(model=model_name)
    elif provider == "openai":
        return model_name
    elif provider == "ollama":
        from deepeval.models import OllamaModel
        return OllamaModel(model=model_name)
    return model_name


def _build_metrics(metric_names: list[str], threshold: float, judge_model_str: str | None) -> list:
    judge = _resolve_judge_model(judge_model_str) if judge_model_str else None
    metrics = []
    for name in metric_names:
        if name == "tool_correctness":
            from deepeval.metrics import ToolCorrectnessMetric
            kwargs = {"threshold": threshold}
            if judge:
                kwargs["model"] = judge
            metrics.append(ToolCorrectnessMetric(**kwargs))
        elif name == "argument_correctness":
            from deepeval.metrics import ArgumentCorrectnessMetric
            kwargs = {"threshold": threshold}
            if judge:
                kwargs["model"] = judge
            metrics.append(ArgumentCorrectnessMetric(**kwargs))
        elif name == "mcp_use":
            from deepeval.metrics import MCPUseMetric
            kwargs = {"threshold": threshold}
            if judge:
                kwargs["model"] = judge
            metrics.append(MCPUseMetric(**kwargs))
        else:
            raise ValueError(f"Unknown metric: {name!r}. Use: tool_correctness, argument_correctness, mcp_use")
    return metrics


def run(
    tool_source,
    test_cases: list[dict],
    model: str,
    repeats: int = 3,
    metric_names: list[str] | None = None,
    threshold: float = 0.7,
    judge_model: str | None = None,
) -> dict:
    if metric_names is None:
        metric_names = ["tool_correctness"]

    tools = asyncio.run(get_tools(tool_source))
    effective_judge = judge_model or model
    metrics = _build_metrics(metric_names, threshold, effective_judge)

    results = []
    all_deepeval_cases = []

    for case in test_cases:
        acceptable = {case["expected_tool"], *case.get("acceptable_tools", [])}
        expected_tools = _build_expected_tools(case)
        runs = []

        for _ in range(repeats):
            tool_calls = call_model(model, case["query"], tools)
            chosen_names = [tc.name for tc in tool_calls]
            runs.append(chosen_names)

            from deepeval.test_case import LLMTestCase, ToolCall

            deepeval_case = LLMTestCase(
                input=case["query"],
                actual_output=f"Called tools: {', '.join(chosen_names) or 'none'}",
                tools_called=[
                    ToolCall(name=tc.name, input_parameters=tc.input_parameters)
                    for tc in tool_calls
                ],
                expected_tools=expected_tools,
            )
            all_deepeval_cases.append((case, deepeval_case))

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

    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

    deepeval_results = evaluate(
        test_cases=[dc for _, dc in all_deepeval_cases],
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            print_results=False,
            show_indicator=False,
            inspect_after_run=False,
        ),
    )

    result_idx = 0
    for r in results:
        scores = []
        for _ in range(repeats):
            if result_idx < len(deepeval_results.test_results):
                test_result = deepeval_results.test_results[result_idx]
                for md in test_result.metrics_data:
                    scores.append(md.score)
            result_idx += 1
        r["deepeval_score"] = sum(scores) / len(scores) if scores else 0.0
        r["deepeval_passed"] = r["deepeval_score"] >= threshold

    total_passes = sum(r["pass_count"] for r in results)
    total_runs = len(results) * repeats
    deepeval_passed = sum(1 for r in results if r["deepeval_passed"])

    return {
        "model": model,
        "repeats": repeats,
        "tool_count": len(tools),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_used": metric_names,
        "threshold": threshold,
        "results": results,
        "accuracy": total_passes / total_runs if total_runs else 0.0,
        "deepeval_pass_rate": deepeval_passed / len(results) if results else 0.0,
    }


def print_table(run_data: dict) -> None:
    repeats = run_data["repeats"]
    has_deepeval = any("deepeval_score" in r for r in run_data["results"])

    if has_deepeval:
        print(f"{'ID':<14}{'PASS':<8}{'SCORE':<8}{'CATEGORY':<20}QUERY")
        for r in run_data["results"]:
            pass_str = f"{r['pass_count']}/{repeats}"
            score_str = f"{r.get('deepeval_score', 0):.2f}"
            print(f"{r['id']:<14}{pass_str:<8}{score_str:<8}{r['category']:<20}{r['query']}")
    else:
        print(f"{'ID':<14}{'PASS':<8}{'CATEGORY':<20}QUERY")
        for r in run_data["results"]:
            pass_str = f"{r['pass_count']}/{repeats}"
            print(f"{r['id']:<14}{pass_str:<8}{r['category']:<20}{r['query']}")

    total_passes = sum(r["pass_count"] for r in run_data["results"])
    total_runs = len(run_data["results"]) * repeats
    print(f"\nTool count: {run_data['tool_count']}")
    print(f"Overall accuracy: {total_passes}/{total_runs} ({run_data['accuracy']:.1%})")
    if has_deepeval:
        print(f"DeepEval pass rate: {run_data.get('deepeval_pass_rate', 0):.1%}")
        print(f"Metrics: {', '.join(run_data.get('metrics_used', []))}")
        print(f"Threshold: {run_data.get('threshold', 0.7)}")


def render_markdown(run_data: dict) -> str:
    repeats = run_data["repeats"]
    results = run_data["results"]
    total_passes = sum(r["pass_count"] for r in results)
    total_runs = len(results) * repeats
    has_deepeval = any("deepeval_score" in r for r in results)

    lines = [
        f"# Tool-Calling Eval: {run_data['model']}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Model | `{run_data['model']}` |",
        f"| Tools available | {run_data['tool_count']} |",
        f"| Test cases | {len(results)} |",
        f"| Repeats | {repeats} |",
        f"| Accuracy | {total_passes}/{total_runs} ({run_data['accuracy']:.1%}) |",
    ]

    if has_deepeval:
        lines.append(f"| DeepEval pass rate | {run_data.get('deepeval_pass_rate', 0):.1%} |")
        lines.append(f"| Metrics | {', '.join(run_data.get('metrics_used', []))} |")
        lines.append(f"| Threshold | {run_data.get('threshold', 0.7)} |")

    lines.append(f"| Timestamp | {run_data['timestamp']} |")
    lines.append("")

    lines.append("## Results")
    lines.append("")

    if has_deepeval:
        lines.append("| ID | Pass | Score | Category | Expected Tool | Query |")
        lines.append("|----|----- |-------|----------|---------------|-------|")
        for r in results:
            pass_str = f"{r['pass_count']}/{repeats}"
            score = f"{r.get('deepeval_score', 0):.2f}"
            status = "PASS" if r.get("deepeval_passed") else "FAIL"
            lines.append(f"| {r['id']} | {pass_str} {status} | {score} | {r['category']} | `{r['expected_tool']}` | {r['query']} |")
    else:
        lines.append("| ID | Pass | Category | Expected Tool | Query |")
        lines.append("|----|------|----------|---------------|-------|")
        for r in results:
            pass_str = f"{r['pass_count']}/{repeats}"
            lines.append(f"| {r['id']} | {pass_str} | {r['category']} | `{r['expected_tool']}` | {r['query']} |")

    failed = [r for r in results if not r.get("deepeval_passed", r["pass_count"] == repeats)]
    if failed:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r['id']}**: expected `{r['expected_tool']}`, got `{', '.join(r['runs'][0]) if r['runs'] else 'none'}`")

    lines.extend(_build_recommendations(results, repeats))

    lines.append("")
    return "\n".join(lines)


def _build_recommendations(results: list[dict], repeats: int) -> list[str]:
    """Analyze failure patterns and generate actionable recommendations."""
    from collections import Counter

    failed = [r for r in results if r["pass_count"] < repeats]
    if not failed:
        return ["", "## Recommendations", "", "No failures detected."]

    lines = ["", "## Recommendations", ""]

    # --- Pattern 1: list vs retrieve confusion ---
    list_retrieve_pairs = []
    for r in failed:
        expected = r["expected_tool"]
        got = r["runs"][0] if r["runs"] else []
        if expected.endswith("_retrieve"):
            base = expected.rsplit("_retrieve", 1)[0]
            list_variant = f"{base}_list"
            if list_variant in got:
                list_retrieve_pairs.append((expected, list_variant))
        elif expected.endswith("_list"):
            base = expected.rsplit("_list", 1)[0]
            retrieve_variant = f"{base}_retrieve"
            if retrieve_variant in got:
                list_retrieve_pairs.append((expected, retrieve_variant))

    if list_retrieve_pairs:
        tools = sorted(set(f"`{a}` vs `{b}`" for a, b in list_retrieve_pairs))
        lines.append(f"### List vs Retrieve Confusion ({len(list_retrieve_pairs)} failures)")
        lines.append("")
        lines.append("The model confuses `_list` and `_retrieve` variants of the same resource. "
                      "Queries without a specific ID tend to trigger the list endpoint instead of retrieve.")
        lines.append("")
        for pair in tools:
            lines.append(f"- {pair}")
        lines.append("")
        lines.append("**Recommendation:** Improve tool descriptions to clearly differentiate "
                      "\"list all items\" vs \"get a single item by ID\". Consider merging into a single "
                      "tool that accepts an optional ID parameter.")
        lines.append("")

    # --- Pattern 2: similar/duplicate tools ---
    confused_with = Counter()
    for r in failed:
        expected = r["expected_tool"]
        got = r["runs"][0] if r["runs"] else []
        for tool_name in got:
            if tool_name != expected:
                pair = tuple(sorted([expected, tool_name]))
                confused_with[pair] += 1

    duplicate_pairs = [(pair, count) for pair, count in confused_with.items()
                       if count >= 2 and pair not in {tuple(sorted(p)) for p in list_retrieve_pairs}]
    if duplicate_pairs:
        duplicate_pairs.sort(key=lambda x: -x[1])
        lines.append(f"### Frequently Confused Tool Pairs ({len(duplicate_pairs)} pairs)")
        lines.append("")
        lines.append("These tool pairs are repeatedly mixed up, suggesting their names or descriptions overlap.")
        lines.append("")
        for (a, b), count in duplicate_pairs:
            lines.append(f"- `{a}` / `{b}` ({count} mix-ups)")
        lines.append("")
        lines.append("**Recommendation:** Differentiate tool descriptions, or consider consolidating "
                      "tools that serve nearly the same purpose.")
        lines.append("")

    # --- Pattern 3: no tool called ---
    no_tool = [r for r in failed if r["runs"] and not r["runs"][0]]
    if no_tool:
        lines.append(f"### No Tool Selected ({len(no_tool)} failures)")
        lines.append("")
        lines.append("The model returned no tool call at all for these queries.")
        lines.append("")
        for r in no_tool:
            lines.append(f"- **{r['id']}** (`{r['expected_tool']}`): \"{r['query']}\"")
        lines.append("")
        lines.append("**Recommendation:** These queries may be too vague or use everyday language "
                      "that doesn't match tool descriptions. Rephrase test queries or improve tool descriptions "
                      "to cover common phrasings.")
        lines.append("")

    # --- Pattern 4: easy vs hard breakdown ---
    easy = [r for r in results if r.get("category") == "easy"]
    hard = [r for r in results if r.get("category") == "hard"]
    if easy and hard:
        easy_rate = sum(r["pass_count"] for r in easy) / (len(easy) * repeats) if easy else 0
        hard_rate = sum(r["pass_count"] for r in hard) / (len(hard) * repeats) if hard else 0
        lines.append("### Accuracy by Category")
        lines.append("")
        lines.append(f"| Category | Tests | Accuracy |")
        lines.append(f"|----------|-------|----------|")
        lines.append(f"| easy | {len(easy)} | {easy_rate:.1%} |")
        lines.append(f"| hard | {len(hard)} | {hard_rate:.1%} |")
        lines.append("")
        if hard_rate < easy_rate - 0.15:
            lines.append(f"**Recommendation:** Hard queries drop {easy_rate - hard_rate:.0%} below easy queries. "
                         "The model struggles with paraphrased or indirect queries. Consider adding synonyms "
                         "and alternate phrasings to tool descriptions.")
            lines.append("")

    # --- Pattern 5: worst tools ---
    from collections import defaultdict
    tool_failures = defaultdict(int)
    tool_totals = defaultdict(int)
    for r in results:
        tool_totals[r["expected_tool"]] += 1
        if r["pass_count"] < repeats:
            tool_failures[r["expected_tool"]] += 1

    worst = [(tool, tool_failures[tool], tool_totals[tool])
             for tool in tool_failures
             if tool_failures[tool] == tool_totals[tool] and tool_totals[tool] >= 2]
    if worst:
        worst.sort(key=lambda x: -x[1])
        lines.append(f"### Consistently Failing Tools ({len(worst)} tools)")
        lines.append("")
        lines.append("These tools failed every test case — the model never selects them.")
        lines.append("")
        for tool, fails, total in worst:
            lines.append(f"- `{tool}` (0/{total} passed)")
        lines.append("")
        lines.append("**Recommendation:** These tools may have poor discoverability. "
                      "Review their names and descriptions for clarity, or consider whether "
                      "they should be consolidated with similar tools.")
        lines.append("")

    return lines


def save_results(run_data: dict, server_name: str, fmt: str = "json") -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_model = run_data["model"].replace(":", "-")
    timestamp = run_data["timestamp"].replace(":", "-")
    base = f"{server_name}_{safe_model}_{timestamp}"

    if fmt == "md":
        path = RESULTS_DIR / f"{base}.md"
        path.write_text(render_markdown(run_data))
    else:
        path = RESULTS_DIR / f"{base}.json"
        path.write_text(json.dumps(run_data, indent=2))
    return path
