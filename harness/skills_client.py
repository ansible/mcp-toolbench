"""Reads an MCP server's skills the way a client has to: through `instructions`.

SEP-2640 serves a skill's `SKILL.md` and, under the same prefix, any supporting
file it cites. `resources/list` advertises only the `SKILL.md`, so a supporting
file is reachable but invisible — a client finds it only because the skill's own
text names it. That makes citation resolution, not listing, the thing to check.

The server states the resolution rule in its `instructions`: take the skill's
URI and replace the final `SKILL.md` segment with the path the skill gives you.
`resolve_citation` applies exactly that rule, so an unresolvable citation here is
one a conforming client could not resolve either.
"""

import posixpath
import re
from dataclasses import dataclass, field

import yaml

SKILL_URI = re.compile(r"skill://[^\s)\]]+/SKILL\.md")
CONTEXT_URI = re.compile(r"aap://context/[^\s)\]]+")

# `[text](target)` with a target that is not a URL, and bare `.md` code spans.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
MD_SPAN = re.compile(r"`([^`\n]*\.md)`")


@dataclass
class Citation:
    """A supporting file a skill's body points at."""

    raw: str
    uri: str | None = None
    note: str = ""


@dataclass
class Skill:
    name: str
    uri: str
    frontmatter: dict
    body: str
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
    context_digests: dict[str, str]
    extension_advertised: bool


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter, body). A skill without frontmatter yields ({}, text)."""
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
    """Pull the two tool declarations a skill carries.

    `allowed-tools` is the agentskills.io field, a whitespace-separated string;
    `mcpToolDependencies` is the Red Hat field, a list of per-server lists. They
    are meant to say the same thing, and a check below asserts they do.
    """
    raw_allowed = frontmatter.get("allowed-tools") or ""
    allowed = set(raw_allowed.split()) if isinstance(raw_allowed, str) else set()

    dependency: set[str] = set()
    for dep in frontmatter.get("mcpToolDependencies") or []:
        if isinstance(dep, dict):
            dependency |= set(dep.get("tools") or [])

    return allowed, dependency


def extract_citations(body: str) -> list[str]:
    """Find the supporting files a skill's body tells the reader to open.

    Markdown links count whenever the target is not a URL. Code spans count only
    when they name a `.md` file, excluding `SKILL.md` (a skill referring to its
    own kind) and anything with a `<placeholder>`, which is an output path the
    skill writes rather than one it reads.
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

    seen: set[str] = set()
    return [c for c in found if not (c in seen or seen.add(c))]


def resolve_citation(skill_uri: str, raw: str) -> Citation:
    """Apply the server's documented rule: swap `SKILL.md` for the cited path.

    A path that climbs out of the skill's own directory has no URI under the
    skill prefix. When it points into the shared-context tree, the note names the
    `aap://context/` address that would work, because that is the fix.
    """
    if raw.startswith(("skill://", "aap://")):
        return Citation(raw=raw, uri=raw)

    # `skill://aap/<name>/SKILL.md` — `aap` is the authority, not a path segment,
    # so climbing past `<name>` leaves the URI space rather than landing on `aap/`.
    scheme, _, rest = skill_uri.partition("://")
    authority, _, path = rest.partition("/")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(path), raw))

    if joined.startswith(".."):
        note = "climbs above the skill root"
        _, sep, tail = raw.partition("context/")
        if sep:
            note += f"; served at aap://context/{tail}"
        return Citation(raw=raw, uri=None, note=note)

    return Citation(raw=raw, uri=f"{scheme}://{authority}/{joined}")


async def gather(transport) -> Snapshot:
    """Connect once and collect instructions, skills, tools and resources."""
    from fastmcp import Client

    async with Client(transport) as client:
        init = client.initialize_result
        instructions = init.instructions or ""
        extensions = getattr(init.capabilities, "extensions", None) or {}

        tool_names = {tool.name for tool in await client.list_tools()}

        resource_uris: set[str] = set()
        context_digests: dict[str, str] = {}
        skill_uris: list[str] = []

        for resource in await client.list_resources():
            uri = str(resource.uri)
            resource_uris.add(uri)
            meta = getattr(resource, "_meta", None) or getattr(resource, "meta", None)
            if isinstance(meta, dict) and meta.get("digest"):
                context_digests[uri] = meta["digest"]
            if uri.startswith("skill://") and uri.endswith("/SKILL.md"):
                skill_uris.append(uri)

        skills = []
        for uri in sorted(skill_uris):
            text = (await client.read_resource(uri))[0].text
            frontmatter, body = split_frontmatter(text)
            allowed, dependency = declared_tools(frontmatter)
            skills.append(
                Skill(
                    name=uri.rsplit("/", 2)[-2],
                    uri=uri,
                    frontmatter=frontmatter,
                    body=body,
                    allowed_tools=allowed,
                    dependency_tools=dependency,
                    citations=[
                        resolve_citation(uri, raw) for raw in extract_citations(body)
                    ],
                )
            )

    return Snapshot(
        instructions=instructions,
        skills=skills,
        tool_names=tool_names,
        resource_uris=resource_uris,
        context_digests=context_digests,
        extension_advertised="io.modelcontextprotocol/skills" in extensions,
    )


async def readable(transport, uris: list[str]) -> dict[str, str | None]:
    """Map each URI to an error string, or None when it reads cleanly."""
    from fastmcp import Client

    outcomes: dict[str, str | None] = {}
    async with Client(transport) as client:
        for uri in uris:
            try:
                await client.read_resource(uri)
                outcomes[uri] = None
            except Exception as exc:
                outcomes[uri] = str(exc)
    return outcomes
