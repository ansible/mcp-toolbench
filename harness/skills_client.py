"""Reads an MCP server's skills the way a client has to: through `instructions`.

`resources/list` advertises only each skill's `SKILL.md`, so the supporting files
it cites are reachable but invisible — which makes citation resolution, not
listing, the thing to check.
"""

import posixpath
import re
from dataclasses import dataclass, field

import yaml

from harness.mcp_client import mcp_session

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
MD_SPAN = re.compile(r"`([^`\n]*\.md)`")

SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


@dataclass
class Citation:
    raw: str
    uri: str | None = None
    note: str = ""


@dataclass
class Skill:
    name: str
    uri: str
    frontmatter: dict
    allowed_tools: set[str] = field(default_factory=set)
    dependency_tools: set[str] = field(default_factory=set)
    citations: list[Citation] = field(default_factory=list)

    @property
    def declared_tools(self) -> set[str]:
        return self.allowed_tools | self.dependency_tools


@dataclass
class Snapshot:
    """Everything the eval needs from one connection to the server."""

    instructions: str
    skills: list[Skill]
    tool_names: set[str]
    resource_uris: set[str]
    reads: dict[str, str | None]
    extension_advertised: bool


def split_frontmatter(text: str) -> tuple[dict, str]:
    """No frontmatter yields ({}, text)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        parsed = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}, parts[2]
    return (parsed if isinstance(parsed, dict) else {}), parts[2]


def declared_tools(frontmatter: dict) -> tuple[set[str], set[str]]:
    """Two competing vocabularies: `allowed-tools` (agentskills.io, whitespace
    string) and `mcpToolDependencies` (Red Hat, per-server lists). A check
    asserts they agree.
    """
    raw_allowed = frontmatter.get("allowed-tools") or ""
    allowed = set(raw_allowed.split()) if isinstance(raw_allowed, str) else set()

    dependency: set[str] = set()
    for dep in frontmatter.get("mcpToolDependencies") or []:
        if isinstance(dep, dict):
            dependency |= set(dep.get("tools") or [])

    return allowed, dependency


def extract_citations(body: str) -> list[str]:
    """Supporting files a skill tells the reader to open.

    Skips `SKILL.md` (a skill naming its own kind) and `<placeholder>` paths,
    which are files the skill writes rather than reads.
    """
    found: list[str] = []

    for target in MD_LINK.findall(body):
        if not target.startswith(("http://", "https://", "#")):
            found.append(target.strip())

    for span in MD_SPAN.findall(body):
        span = span.strip()
        if span == "SKILL.md" or "<" in span:
            continue
        found.append(span)

    return list(dict.fromkeys(found))


def resolve_citation(skill_uri: str, raw: str, served: set[str] = frozenset()) -> Citation:
    """Apply the server's documented rule: swap `SKILL.md` for the cited path.

    A path climbing out of the skill's directory has no URI under the skill
    prefix; if the server serves that file elsewhere, the note names it.
    """
    if SCHEME.match(raw):
        return Citation(raw=raw, uri=raw)

    # In `skill://aap/<name>/SKILL.md`, `aap` is the authority, not a path
    # segment, so climbing past `<name>` leaves the URI space entirely.
    scheme, _, rest = skill_uri.partition("://")
    authority, _, path = rest.partition("/")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(path), raw))

    if joined.startswith(".."):
        note = "climbs above the skill root"
        relative = "/".join(s for s in joined.split("/") if s not in ("..", "."))
        elsewhere = next((u for u in sorted(served) if u.endswith(relative)), None)
        if elsewhere:
            note += f"; served at {elsewhere}"
        return Citation(raw=raw, uri=None, note=note)

    return Citation(raw=raw, uri=f"{scheme}://{authority}/{joined}")


async def gather(transport) -> Snapshot:
    """Connect once and collect instructions, skills, tools and read outcomes."""
    async with mcp_session(transport) as client:
        init = client.initialize_result
        instructions = init.instructions or ""
        extensions = getattr(init.capabilities, "extensions", None) or {}

        tool_names = {tool.name for tool in await client.list_tools()}
        resource_uris = {str(r.uri) for r in await client.list_resources()}
        skill_uris = sorted(
            u for u in resource_uris
            if u.startswith("skill://") and u.endswith("/SKILL.md")
        )

        # SKILL.md reads here, so it needs no second attempt below.
        reads: dict[str, str | None] = {}
        skills = []
        for uri in skill_uris:
            text = (await client.read_resource(uri))[0].text
            reads[uri] = None
            frontmatter, body = split_frontmatter(text)
            allowed, dependency = declared_tools(frontmatter)
            skills.append(
                Skill(
                    name=uri.rsplit("/", 2)[-2],
                    uri=uri,
                    frontmatter=frontmatter,
                    allowed_tools=allowed,
                    dependency_tools=dependency,
                    citations=[
                        resolve_citation(uri, raw, resource_uris)
                        for raw in extract_citations(body)
                    ],
                )
            )

        cited = {c.uri for s in skills for c in s.citations if c.uri} - reads.keys()
        for uri in sorted(cited):
            try:
                await client.read_resource(uri)
                reads[uri] = None
            except Exception as exc:
                reads[uri] = str(exc)

    return Snapshot(
        instructions=instructions,
        skills=skills,
        tool_names=tool_names,
        resource_uris=resource_uris,
        reads=reads,
        extension_advertised="io.modelcontextprotocol/skills" in extensions,
    )
