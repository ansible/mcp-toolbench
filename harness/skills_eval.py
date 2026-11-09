"""Checks that a server's skills are correctly constructed, and that a model uses them.

**Construction** is static — no model, no tokens: could each skill be followed
at all? **Use** is agentic, giving the model only what a conforming client would
(the server's `instructions` as system prompt, plus `read_resource`) and checking
what it does against the skill's own declarations. Every check is a protocol
event, never a judgement about prose.

`read_resource` is us standing in for a client, since none implements SEP-2640
natively yet: results say whether a correct client *would* reach the skills, not
what any shipping product does.
"""

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from harness.runner import run_session
from harness.skills_client import Skill, Snapshot, gather

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


def _check(skill: str, name: str, ok: bool, when_ok: str, when_bad: str) -> Check:
    return Check(skill=skill, check=name, passed=ok, detail=when_ok if ok else when_bad)


class ResourceReader:
    """Answers `read_resource` itself and records every URI the model asked for.

    It is the client's own capability under SEP-2640, not a tool the server
    offers, so it is intercepted rather than forwarded.
    """

    def __init__(self):
        self.uris: list[str] = []

    async def __call__(self, client, call):
        if call.name != READ_RESOURCE_TOOL["name"]:
            return None
        uri = call.input_parameters.get("uri", "")
        self.uris.append(uri)
        return (await client.read_resource(uri))[0].text


def is_write_tool(name: str) -> bool:
    return name.endswith(WRITE_SUFFIXES)


def check_construction(snapshot: Snapshot) -> list[Check]:
    """Everything that can be known about a skill without asking a model."""
    checks = [
        _check(
            "(server)", "extension-advertised", snapshot.extension_advertised,
            "io.modelcontextprotocol/skills in capabilities.extensions",
            "io.modelcontextprotocol/skills absent from capabilities.extensions",
        ),
        _check(
            "(server)", "instructions-present", bool(snapshot.instructions.strip()),
            f"{len(snapshot.instructions)} chars", "empty",
        ),
    ]

    for skill in snapshot.skills:
        name = skill.name
        fm_name = skill.frontmatter.get("name")
        missing = sorted(skill.declared_tools - snapshot.tool_names)
        writes = sorted(t for t in skill.declared_tools if is_write_tool(t))
        unresolved = [
            f"{c.raw} ({c.note})" if c.uri is None else f"{c.raw} -> {c.uri}: unreadable"
            for c in skill.citations
            if c.uri is None or snapshot.reads.get(c.uri) is not None
        ]

        checks += [
            _check(
                name, "advertised", skill.uri in snapshot.instructions,
                "named in the instructions pointer",
                "absent from the instructions pointer",
            ),
            _check(
                name, "frontmatter-name", fm_name == name,
                f"frontmatter name={fm_name!r}",
                f"frontmatter name={fm_name!r}, directory={name!r}",
            ),
            _check(
                name, "declarations-agree",
                skill.allowed_tools == skill.dependency_tools,
                "allowed-tools matches mcpToolDependencies",
                f"differ by {sorted(skill.allowed_tools ^ skill.dependency_tools)}",
            ),
            _check(
                name, "declared-tools-exist", not missing,
                f"{len(skill.declared_tools)} declared, all served",
                f"not served by this endpoint: {missing}",
            ),
            _check(
                name, "declares-no-writes", not writes,
                "read-only declaration", f"declares write tools: {writes}",
            ),
            _check(
                name, "citations-resolve", not unresolved,
                f"{len(skill.citations)} citations, all readable",
                "; ".join(unresolved),
            ),
        ]

    return checks


def expected_reads(skill: Skill, case: dict, reads: dict) -> list[str]:
    """The skill, plus every citation that resolves and reads.

    A broken citation is already a construction failure; counting it here too
    would charge one defect twice. `expected_resources` narrows this when only
    part of a skill's procedure applies to the query.
    """
    if skill is None:
        return []
    if "expected_resources" in case:
        return [skill.uri] + list(case["expected_resources"])
    return [skill.uri] + [
        c.uri for c in skill.citations if c.uri and reads.get(c.uri) is None
    ]


def check_use(skill, case, called, reads, state, expected) -> list[Check]:
    name = case["skill"]
    skill_uri = skill.uri if skill else ""
    declared = skill.declared_tools if skill else set()
    unread = [uri for uri in expected if uri not in reads]
    undeclared = sorted({t for t in called if t not in declared})
    writes = sorted({t for t in called if is_write_tool(t)})

    checks = [
        _check(
            name, "reached", skill_uri in reads,
            f"read {skill_uri}",
            f"never read the skill; read {reads or 'nothing'}",
        ),
        _check(
            name, "followed-citations", not unread,
            f"read all {len(expected)} expected resources",
            f"did not read: {unread}",
        ),
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
        ),
        _check(
            name, "no-writes", not writes,
            "no write tools called", f"wrote: {writes}",
        ),
        Check(
            skill=name,
            check="completed",
            passed=state.finished and bool(called),
            detail=(
                f"finished in {state.turns} turns"
                if state.finished and called
                else "stopped without calling a tool — asked the operator to confirm scope?"
                if state.finished
                else f"exhausted {state.turns} turns"
            ),
        ),
    ]

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
    by_name = {s.name: s for s in snapshot.skills}

    construction = check_construction(snapshot)
    print(f"\nConstruction: {len(snapshot.skills)} skills, {len(construction)} checks")
    for check in construction:
        if not check.passed:
            print(f"  ✗ {check.skill}: {check.check} — {check.detail}", flush=True)

    use: list[dict] = []
    if not construction_only:
        for i, case in enumerate(test_cases, 1):
            print(f"\n  [{i}/{len(test_cases)}] {case['id']} ({case['skill']})", flush=True)
            skill = by_name.get(case["skill"])
            expected = expected_reads(skill, case, snapshot.reads)

            for r in range(repeats):
                prefix = f"    [repeat {r + 1}/{repeats}]"
                print(prefix, flush=True)

                reader = ResourceReader()
                tool_calls, state = run_session(
                    transport, case["query"], model, max_turns,
                    extra_tools=[READ_RESOURCE_TOOL],
                    system=snapshot.instructions,
                    intercept=reader,
                    verbose_prefix=prefix,
                )
                called = [
                    c.name for c in tool_calls if c.name != READ_RESOURCE_TOOL["name"]
                ]

                checks = check_use(skill, case, called, reader.uris, state, expected)
                for check in checks:
                    mark = "✓" if check.passed else "✗"
                    print(f"{prefix}  {mark} {check.check} — {check.detail}", flush=True)

                use.append(
                    {
                        "id": case["id"],
                        "skill": case["skill"],
                        "repeat": r + 1,
                        "query": case["query"],
                        "tools_called": called,
                        "resources_read": reader.uris,
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


def _use_rows(run_data: dict):
    for entry in run_data["use"]:
        for check in entry["checks"]:
            yield f"{entry['id']}#{entry['repeat']}", check


def print_table(run_data: dict) -> None:
    print(f"\n{'SKILL':<34}{'CHECK':<24}{'':<6}DETAIL")
    for check in run_data["construction"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"{check['skill']:<34}{check['check']:<24}{mark:<6}{check['detail']}")

    if run_data["use"]:
        print()
        for label, check in _use_rows(run_data):
            mark = "PASS" if check["passed"] else "FAIL"
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
        for label, check in _use_rows(run_data):
            mark = "PASS" if check["passed"] else "**FAIL**"
            lines.append(f"| {label} | {check['check']} | {mark} | {check['detail']} |")

    failures = [c for c in run_data["construction"] if not c["passed"]]
    if failures:
        lines += ["", "## Construction failures", ""]
        for check in failures:
            lines.append(f"- **{check['skill']}** `{check['check']}`: {check['detail']}")

    lines.append("")
    return "\n".join(lines)
