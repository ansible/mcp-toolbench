"""Checks that a server's skills are correctly constructed, and that a model uses them.

Two passes over one server:

**Construction** is static — no model, no tokens, seconds to run. It asks whether
each skill could be followed at all: is it advertised where a client will look,
do its tool declarations name tools that exist, does every file it cites resolve
to a readable URI.

**Use** is agentic. The model gets the server's `instructions` as its system
prompt and a `read_resource` tool, and nothing else — the same two things a
conforming client would give it. What it does with them is then checked against
the skill's own declarations rather than against a rubric, so every result here
is a protocol event, not a judgement about prose.

Note that the `read_resource` tool is us standing in for a client: no mainstream
client implements SEP-2640 natively yet. These results say whether a correct
client *would* reach the skills, not what any particular product does today.
"""

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from harness.mcp_client import mcp_session, normalize_tools
from harness.runner import run_turns, save_results as _save
from harness.skills_client import Snapshot, gather, readable

WRITE_SUFFIXES = ("_create", "_update", "_partial_update", "_destroy", "_delete")

READ_RESOURCE_TOOL = {
    "name": "read_resource",
    "description": (
        "Read an MCP resource by URI, such as a skill's SKILL.md or one of its "
        "supporting files. Use this to follow the skills named in the server "
        "instructions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "The resource URI to read."}
        },
        "required": ["uri"],
    },
}


@dataclass
class Check:
    skill: str
    check: str
    passed: bool
    detail: str = ""


def is_write_tool(name: str) -> bool:
    return name.endswith(WRITE_SUFFIXES)


def check_construction(snapshot: Snapshot, reads: dict[str, str | None]) -> list[Check]:
    """Everything that can be known about a skill without asking a model."""
    checks: list[Check] = []

    checks.append(
        Check(
            skill="(server)",
            check="extension-advertised",
            passed=snapshot.extension_advertised,
            detail="io.modelcontextprotocol/skills in capabilities.extensions",
        )
    )
    checks.append(
        Check(
            skill="(server)",
            check="instructions-present",
            passed=bool(snapshot.instructions.strip()),
            detail=f"{len(snapshot.instructions)} chars",
        )
    )

    for skill in snapshot.skills:
        name = skill.name

        checks.append(
            Check(
                skill=name,
                check="advertised",
                passed=skill.uri in snapshot.instructions,
                detail="named in the instructions pointer",
            )
        )

        fm_name = skill.frontmatter.get("name")
        checks.append(
            Check(
                skill=name,
                check="frontmatter-name",
                passed=fm_name == name,
                detail=f"frontmatter name={fm_name!r}, directory={name!r}",
            )
        )

        agree = skill.allowed_tools == skill.dependency_tools
        disagreement = skill.allowed_tools ^ skill.dependency_tools
        checks.append(
            Check(
                skill=name,
                check="declarations-agree",
                passed=agree,
                detail=(
                    "allowed-tools matches mcpToolDependencies"
                    if agree
                    else f"differ by {sorted(disagreement)}"
                ),
            )
        )

        missing_tools = sorted(skill.declared_tools - snapshot.tool_names)
        checks.append(
            Check(
                skill=name,
                check="declared-tools-exist",
                passed=not missing_tools,
                detail=(
                    f"{len(skill.declared_tools)} declared, all served"
                    if not missing_tools
                    else f"not served by this endpoint: {missing_tools}"
                ),
            )
        )

        declared_writes = sorted(t for t in skill.declared_tools if is_write_tool(t))
        checks.append(
            Check(
                skill=name,
                check="declares-no-writes",
                passed=not declared_writes,
                detail=(
                    "read-only declaration"
                    if not declared_writes
                    else f"declares write tools: {declared_writes}"
                ),
            )
        )

        failures = []
        for citation in skill.citations:
            if citation.uri is None:
                failures.append(f"{citation.raw} ({citation.note})")
            elif reads.get(citation.uri) is not None:
                failures.append(f"{citation.raw} -> {citation.uri}: unreadable")
        checks.append(
            Check(
                skill=name,
                check="citations-resolve",
                passed=not failures,
                detail=(
                    f"{len(skill.citations)} citations, all readable"
                    if not failures
                    else "; ".join(failures)
                ),
            )
        )

    return checks


def _expected_reads(snapshot: Snapshot, case: dict, reads: dict) -> list[str]:
    """What the model must read: the skill, plus whatever the case pins down.

    Defaults to every citation that both resolves and reads, so a skill is held
    to its own text. A citation that does neither is already a construction
    failure, and counting it here too would charge one defect twice. A case can
    narrow this with `expected_resources` when only part of a skill's procedure
    applies to the query.
    """
    skill = next((s for s in snapshot.skills if s.name == case["skill"]), None)
    if skill is None:
        return []
    if "expected_resources" in case:
        return [skill.uri] + list(case["expected_resources"])
    return [skill.uri] + [
        c.uri for c in skill.citations if c.uri and reads.get(c.uri) is None
    ]


def _run_use_case(transport, snapshot, case, model, max_turns, prefix=""):
    """One agentic run with `instructions` as the system prompt.

    `read_resource` is intercepted rather than forwarded: it is the client's own
    capability under SEP-2640, not a tool the server offers, and intercepting it
    is what records which URIs the model actually reached for.
    """
    reads: list[str] = []

    async def _loop():
        async with mcp_session(transport) as client:
            tools = normalize_tools(await client.list_tools()) + [READ_RESOURCE_TOOL]

            async def intercept(call):
                if call.name != "read_resource":
                    return None
                uri = call.input_parameters.get("uri", "")
                reads.append(uri)
                return (await client.read_resource(uri))[0].text

            return await run_turns(
                client,
                model,
                case["query"],
                tools,
                max_turns,
                system=snapshot.instructions,
                intercept=intercept,
                verbose_prefix=prefix,
            )

    tool_calls, state = asyncio.run(_loop())
    return tool_calls, reads, state


def check_use(snapshot, case, tool_calls, reads, state, readable_uris) -> list[Check]:
    name = case["skill"]
    skill = next((s for s in snapshot.skills if s.name == name), None)
    called = [c.name for c in tool_calls if c.name != "read_resource"]
    expected = _expected_reads(snapshot, case, readable_uris)
    skill_uri = skill.uri if skill else ""

    checks = [
        Check(
            skill=name,
            check="reached",
            passed=skill_uri in reads,
            detail=(
                f"read {skill_uri}"
                if skill_uri in reads
                else f"never read the skill; read {reads or 'nothing'}"
            ),
        )
    ]

    unread = [uri for uri in expected if uri not in reads]
    checks.append(
        Check(
            skill=name,
            check="followed-citations",
            passed=not unread,
            detail=(
                f"read all {len(expected)} expected resources"
                if not unread
                else f"did not read: {unread}"
            ),
        )
    )

    declared = skill.declared_tools if skill else set()
    undeclared = sorted({t for t in called if t not in declared})
    checks.append(
        Check(
            skill=name,
            check="tools-declared",
            passed=not undeclared or not declared,
            detail=(
                "no tool declaration to check against"
                if not declared
                else f"{len(set(called))} tools called, all declared"
                if not undeclared
                else f"called but not declared: {undeclared}"
            ),
        )
    )

    writes = sorted({t for t in called if is_write_tool(t)})
    checks.append(
        Check(
            skill=name,
            check="no-writes",
            passed=not writes,
            detail="no write tools called" if not writes else f"wrote: {writes}",
        )
    )

    completed = state.finished and bool(called)
    checks.append(
        Check(
            skill=name,
            check="completed",
            passed=completed,
            detail=(
                f"finished in {state.turns} turns"
                if completed
                else "stopped without calling a tool — asked the operator to confirm scope?"
                if state.finished
                else f"exhausted {state.turns} turns"
            ),
        )
    )

    return checks


def run_skills(
    transport,
    test_cases: list[dict],
    model: str,
    repeats: int = 1,
    max_turns: int = 15,
    threshold: float = 1.0,
    construction_only: bool = False,
) -> dict:
    snapshot = asyncio.run(gather(transport))

    cited = sorted(
        {c.uri for s in snapshot.skills for c in s.citations if c.uri}
        | {s.uri for s in snapshot.skills}
    )
    reads = asyncio.run(readable(transport, cited))

    construction = check_construction(snapshot, reads)
    print(f"\nConstruction: {len(snapshot.skills)} skills, {len(construction)} checks")
    for check in construction:
        if not check.passed:
            print(f"  ✗ {check.skill}: {check.check} — {check.detail}", flush=True)

    use: list[dict] = []
    if not construction_only:
        for i, case in enumerate(test_cases, 1):
            print(f"\n  [{i}/{len(test_cases)}] {case['id']} ({case['skill']})", flush=True)
            for r in range(repeats):
                prefix = f"    [repeat {r + 1}/{repeats}]"
                print(prefix, flush=True)
                tool_calls, reads_seen, state = _run_use_case(
                    transport, snapshot, case, model, max_turns, prefix
                )
                checks = check_use(snapshot, case, tool_calls, reads_seen, state, reads)
                for check in checks:
                    mark = "✓" if check.passed else "✗"
                    print(f"{prefix}  {mark} {check.check} — {check.detail}", flush=True)
                use.append(
                    {
                        "id": case["id"],
                        "skill": case["skill"],
                        "repeat": r + 1,
                        "query": case["query"],
                        "tools_called": [c.name for c in tool_calls if c.name != "read_resource"],
                        "resources_read": reads_seen,
                        "turns": state.turns,
                        "checks": [asdict(c) for c in checks],
                    }
                )

    construction_passed = sum(1 for c in construction if c.passed)
    use_checks = [c for entry in use for c in entry["checks"]]
    use_passed = sum(1 for c in use_checks if c["passed"])
    total = len(construction) + len(use_checks)

    return {
        "model": model,
        "mode": "skills",
        "repeats": repeats,
        "max_turns": max_turns,
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skill_count": len(snapshot.skills),
        "tool_count": len(snapshot.tool_names),
        "construction": [asdict(c) for c in construction],
        "construction_accuracy": construction_passed / len(construction) if construction else 0.0,
        "use": use,
        "use_accuracy": use_passed / len(use_checks) if use_checks else 0.0,
        "accuracy": (construction_passed + use_passed) / total if total else 0.0,
    }


def _group(checks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for check in checks:
        grouped.setdefault(check["skill"], []).append(check)
    return grouped


def print_table(run_data: dict) -> None:
    print(f"\n{'SKILL':<34}{'CHECK':<24}{'':<6}DETAIL")
    for skill, checks in _group(run_data["construction"]).items():
        for check in checks:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"{skill:<34}{check['check']:<24}{mark:<6}{check['detail']}")

    if run_data["use"]:
        print()
        for entry in run_data["use"]:
            for check in entry["checks"]:
                mark = "PASS" if check["passed"] else "FAIL"
                label = f"{entry['id']}#{entry['repeat']}"
                print(f"{label:<34}{check['check']:<24}{mark:<6}{check['detail']}")

    print(f"\nSkills: {run_data['skill_count']}   Tools: {run_data['tool_count']}")
    print(f"Construction: {run_data['construction_accuracy']:.1%}")
    if run_data["use"]:
        print(f"Use:          {run_data['use_accuracy']:.1%}")
    print(f"Overall:      {run_data['accuracy']:.1%}")


def render_markdown(run_data: dict) -> str:
    lines = [
        f"# Skills Eval: {run_data['model']}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Skills | {run_data['skill_count']} |",
        f"| Tools | {run_data['tool_count']} |",
        f"| Construction | {run_data['construction_accuracy']:.1%} |",
        f"| Use | {run_data['use_accuracy']:.1%} |",
        f"| Overall | {run_data['accuracy']:.1%} |",
        f"| Timestamp | {run_data['timestamp']} |",
        "",
        "## Construction",
        "",
        "| Skill | Check | Result | Detail |",
        "|-------|-------|--------|--------|",
    ]
    for check in run_data["construction"]:
        mark = "PASS" if check["passed"] else "**FAIL**"
        lines.append(
            f"| `{check['skill']}` | {check['check']} | {mark} | {check['detail']} |"
        )

    if run_data["use"]:
        lines += [
            "",
            "## Use",
            "",
            "| Case | Check | Result | Detail |",
            "|------|-------|--------|--------|",
        ]
        for entry in run_data["use"]:
            for check in entry["checks"]:
                mark = "PASS" if check["passed"] else "**FAIL**"
                label = f"{entry['id']}#{entry['repeat']}"
                lines.append(
                    f"| {label} | {check['check']} | {mark} | {check['detail']} |"
                )

    failures = [c for c in run_data["construction"] if not c["passed"]]
    if failures:
        lines += ["", "## Construction failures", ""]
        for check in failures:
            lines.append(f"- **{check['skill']}** `{check['check']}`: {check['detail']}")

    lines.append("")
    return "\n".join(lines)


def save_results(run_data: dict, eval_name: str, fmt: str = "json") -> Path:
    return _save(run_data, eval_name, fmt=fmt, render=render_markdown)
