"""Tool-calling agent loop that drives OLAF's MCP tools to build an ontology.

Ported from OLAF's own reference agent
(https://github.com/merlin-intelligence/olaf, ``demos/olaf_building_agent/agent.py``),
adapted to:
  - stream progress through a ``log()`` callback into
    :class:`cogmaps.ontology.store.OntologyJobStore` instead of stderr logging,
  - scope the initial user turn to the documents selected on the page,
  - call Scaleway through ``litellm``'s native ``scaleway/`` provider
    (``model="scaleway/<name>"`` + ``api_key``) — the same provider used by
    :mod:`cogmaps.rag.llm_clients`'s ``ScalewayClient``,
  - persist the Turtle via an ``export_cb()`` callback instead of a static
    file path from a toml,
  - take the seeds to reuse from the job (selected on the page) instead of
    ``[olaf] seed_ids``, and list their general classes in the initial user
    turn — this agent reads the chunks itself, so there is no per-batch digest.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from mcp import ClientSession

from cogmaps.core.llm_retry import retrying_completion
from cogmaps.ontology.olaf_config import has_ontology_content
from cogmaps.ontology.prompts import build_system_prompt

if TYPE_CHECKING:
    from cogmaps.ontology.domain_discovery import DomainBlueprint

MAX_ITERATIONS = 150

# OLAF tools hidden from the build agent, and refused if called anyway:
# checking, inferring and deleting entities belong to the reasoning agent
# (cogmaps.ontology.reasoning_agent), run after the build — Pellet is slow, and
# its report would only distract the extraction — and the chunk collection is
# fixed by the job's config.toml. Seeds are loaded from the Ontology Explorer
# page and selected per job, not by the agent.
HIDDEN_TOOLS = frozenset({
    "ontology_check", "ontology_infer", "entity_delete", "chunk_collection_switch", "seed_load", "seed_select",
})

# Only offered when seeds are selected for the build — without any, they can only fail.
SEED_TOOLS = frozenset({"seed_reuse", "seed_term_get"})

# Searches that always cover the seeds selected for the build (OLAF's agent does the same).
SEED_SEARCH_TOOLS = frozenset({"concept_search", "concept_semantic_search", "property_search", "property_semantic_search"})

# How many of the seeds' general classes the initial user turn lists.
MAX_SEED_PARENTS = 30

# General classes of the seeds — those with several subclasses there (LegalEntity, Polity,
# Loan…): the candidate parents of the ontology's new classes. From OLAF's
# demos/olaf_building_agent/agent.py.
_SEED_PARENTS_QUERY = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?p ?g (SAMPLE(?l) AS ?label) (COUNT(DISTINCT ?c) AS ?children) WHERE {
    GRAPH ?g {
        ?c rdfs:subClassOf ?p . ?p a owl:Class .
        FILTER(isIRI(?c) && isIRI(?p) && !STRSTARTS(STR(?p), "http://www.w3.org/"))
        OPTIONAL { ?p rdfs:label|skos:prefLabel ?l }
    }
    VALUES ?g { __SEED_GRAPHS__ }
} GROUP BY ?p ?g HAVING(COUNT(DISTINCT ?c) >= 2) ORDER BY DESC(?children) ?p
"""

# How many times in a row the model may stop while chunks are still pending
# before the loop gives up (each nudge costs a full completion call).
MAX_CONSECUTIVE_NUDGES = 3

# Context compaction: tool results older than the most recent
# KEEP_RECENT_TOOL_RESULTS are cut down to COMPACTED_PREVIEW_CHARS once they
# exceed COMPACT_THRESHOLD_CHARS. Chunk texts (chunk_read_batch) and bulk
# listings dominate the context and are no longer needed once processed;
# short results such as the URIs returned by concept_create stay intact.
KEEP_RECENT_TOOL_RESULTS = 12
COMPACT_THRESHOLD_CHARS = 1500
COMPACTED_PREVIEW_CHARS = 300
_COMPACTED_MARKER = "[…truncated to save context — call the tool again if you need the full result]"


def mcp_tools_to_litellm(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to the OpenAI function-calling format used by LiteLLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def _summarise(args: dict) -> str:
    """Compact one-line representation of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:40]!r}" if len(s) > 40 else f"{k}={s!r}")
    return ", ".join(parts)


def _build_user_message(doc_filenames: list[str], seed_ids: Sequence[str] = (), seed_parents: Sequence[str] = ()) -> str:
    if doc_filenames:
        docs = ", ".join(doc_filenames)
        message = (
            f"Build the ontology from the Qdrant chunks of these {len(doc_filenames)} "
            f"document(s) only: {docs}. Scope every chunk_list call to one of these "
            f"document IDs at a time via the doc_id parameter."
        )
    else:
        message = "Build the ontology from all available Qdrant chunks in the collection."
    if seed_ids:
        message += f"\n\nSeeds selected for this build: {', '.join(seed_ids)}."
        if seed_parents:
            message += (
                "\n\n## General seed classes — candidate parents: what kind of thing is each new class?\n"
                + "\n".join(seed_parents)
            )
    return message


def _local_name(uri: str) -> str:
    return uri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


async def select_seeds(session: ClientSession, seed_ids: list[str], log: Callable[[str], None]) -> list[str]:
    """Restrict the OLAF session to ``seed_ids`` (``seed_select``) — those still loaded.

    Always called, even with no seed: OLAF's default is every seed in the store.
    """
    result = await session.call_tool("seed_list", {})
    loaded = {s["id"] for s in json.loads(result.content[0].text)} if result.content else set()
    if missing := [s for s in seed_ids if s not in loaded]:
        log(f"Warning: seed(s) no longer loaded, skipped: {', '.join(missing)}")
    selected = [s for s in seed_ids if s in loaded]
    result = await session.call_tool("seed_select", {"seed_ids": selected})
    text = result.content[0].text if result.content else ""
    if text.startswith("Error"):
        raise RuntimeError(f"seed_select failed: {text}")
    log(f"Seeds: {', '.join(selected)}" if selected else "Seeds: none")
    return selected


async def _seed_parents(session: ClientSession, seed_ids: list[str], log: Callable[[str], None]) -> list[str]:
    """User-message lines for the selected seeds' general classes (see _SEED_PARENTS_QUERY)."""
    graphs = " ".join(f"<urn:olaf:seed:{s}>" for s in seed_ids)
    query = _SEED_PARENTS_QUERY.replace("__SEED_GRAPHS__", graphs)
    try:
        result = await session.call_tool("sparql_query", {"query": query, "limit": MAX_SEED_PARENTS})
        rows = json.loads(result.content[0].text)["rows"]
    except Exception as exc:  # noqa: BLE001 — the build goes on: the searches still find seed terms
        log(f"Warning: could not list the seeds' general classes: {exc}")
        return []
    return [
        f"- {r['label'] or _local_name(r['p'])} <{r['p']}> [seed {(r['g'] or '').removeprefix('urn:olaf:seed:')}, "
        f"{r['children']} subclasses]"
        for r in rows
    ]


def _compact_history(messages: list[dict[str, Any]]) -> int:
    """Truncate old, large tool results in place. Returns how many were compacted.

    Only ``content`` is shortened — every tool message stays, so each
    assistant ``tool_calls`` entry keeps its matching ``tool_call_id``.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    compacted = 0
    for i in tool_indices[:-KEEP_RECENT_TOOL_RESULTS] if KEEP_RECENT_TOOL_RESULTS else tool_indices:
        content = messages[i].get("content") or ""
        if len(content) > COMPACT_THRESHOLD_CHARS and not content.endswith(_COMPACTED_MARKER):
            messages[i]["content"] = f"{content[:COMPACTED_PREVIEW_CHARS]}\n{_COMPACTED_MARKER}"
            compacted += 1
    return compacted


async def _docs_with_pending_chunks(session: ClientSession, doc_filenames: list[str]) -> list[str] | None:
    """Ask OLAF which of the requested documents still have pending chunks.

    Returns the documents (or ``["<collection>"]`` when the build isn't scoped
    to documents) that still have at least one pending chunk, or None if OLAF
    couldn't answer — the caller then treats the work as unfinished.
    """
    scopes: list[str | None] = list(doc_filenames) or [None]
    remaining: list[str] = []
    for doc in scopes:
        args: dict[str, Any] = {"status": "pending", "limit": 1}
        if doc:
            args["doc_id"] = doc
        try:
            result = await session.call_tool("chunk_list", args)
            chunks = json.loads(result.content[0].text) if result.content else []
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(chunks, list):
            return None
        if chunks:
            remaining.append(doc or "<collection>")
    return remaining


def _nudge_message(remaining: list[str] | None) -> str:
    where = f" Documents with pending chunks: {', '.join(remaining)}." if remaining else ""
    return (
        "Do not stop yet — some chunks are still pending." + where + " "
        "Call chunk_list with doc_id and status=\"pending\" (move on to the next doc_id once a document has none left), "
        "read them in batches of 5 to 10 with chunk_read_batch, extract what is relevant (search for existing "
        "concepts and properties before creating new ones), and mark chunks processed. Once no pending chunks "
        "remain, call ontology_orphans and connect the isolated entities, then call ontology_export."
    )


async def run_build(
    session: ClientSession,
    *,
    model: str,
    api_key: str,
    doc_filenames: list[str],
    log: Callable[[str], None],
    set_progress: Callable[[int, int], None],
    export_cb: Callable[[str], None],
    max_iterations: int = MAX_ITERATIONS,
    blueprint: DomainBlueprint | None = None,
    system_prompt: str | None = None,
    seed_ids: Sequence[str] = (),
) -> None:
    """Run the tool-calling loop against an already-initialized OLAF session.

    ``log(message)`` is called for every notable step (tool call + truncated
    result). ``export_cb(turtle)`` is called with the Turtle content whenever
    the model calls ``ontology_export``. ``set_progress(current, total)``
    tracks loop iterations for the page's progress bar. ``seed_ids`` are the
    seed ontologies the agent may reuse terms from (none if empty).
    """
    seed_ids = await select_seeds(session, list(seed_ids), log)
    hidden = HIDDEN_TOOLS if seed_ids else HIDDEN_TOOLS | SEED_TOOLS
    forced_args = {t: {"include_seeds": True} for t in SEED_SEARCH_TOOLS} if seed_ids else {}

    tools_result = await session.list_tools()
    tools = mcp_tools_to_litellm([t for t in tools_result.tools if t.name not in hidden])
    log(f"Loaded {len(tools)} OLAF tools.")

    active_prompt = system_prompt or build_system_prompt(blueprint, use_seeds=bool(seed_ids))
    if blueprint:
        log(f"Configured agent with dynamic domain blueprint: '{blueprint.inferred_domain}' ({len(blueprint.pillars)} pillars)")
    else:
        log("Configured agent with default domain prompt.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": active_prompt},
        {"role": "user", "content": _build_user_message(
            doc_filenames, seed_ids, await _seed_parents(session, seed_ids, log) if seed_ids else [],
        )},
    ]

    total_tool_calls = 0
    consecutive_nudges = 0

    for iteration in range(1, max_iterations + 1):
        log(f"--- Iteration {iteration}/{max_iterations} ---")
        set_progress(iteration, max_iterations)

        compacted = _compact_history(messages)
        if compacted:
            log(f"Compacted {compacted} old tool result(s) to keep the context small.")

        response = await asyncio.to_thread(
            retrying_completion,
            log,
            model=f"scaleway/{model}",
            api_key=api_key,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            # litellm's "scaleway" provider config doesn't whitelist tools/
            # tool_choice as supported params (as of litellm 1.101.0) even
            # though Scaleway's own API supports tool-calling — without this,
            # litellm raises UnsupportedParamsError instead of forwarding them.
            allowed_openai_params=["tools", "tool_choice"],
            max_tokens=4096,
            temperature=0,
            timeout=120,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            if msg.content:
                log(f"Agent message: {msg.content[:200]}")
            # Stopping is only accepted once OLAF confirms no requested chunk is
            # still pending — the model's own claim that it is done isn't enough.
            remaining = await _docs_with_pending_chunks(session, doc_filenames)
            if remaining == []:
                log("Agent finished — no pending chunks remain.")
                break
            if consecutive_nudges >= MAX_CONSECUTIVE_NUDGES or iteration == max_iterations:
                pending_info = ", ".join(remaining) if remaining else "unknown (chunk_list failed)"
                log(
                    f"Agent stopped after {consecutive_nudges} nudge(s) while chunks are still pending "
                    f"({pending_info}). Re-launch the build to continue — processed chunks are skipped."
                )
                break
            consecutive_nudges += 1
            log(f"Nudging the agent to continue ({consecutive_nudges}/{MAX_CONSECUTIVE_NUDGES}).")
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": _nudge_message(remaining)})
            continue

        consecutive_nudges = 0
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            tool_args = {**tool_args, **forced_args.get(tool_name, {})}

            log(f"Tool call: {tool_name}({_summarise(tool_args)})")
            total_tool_calls += 1

            if tool_name in hidden:
                content = f"Error: {tool_name} is not available while building."
            else:
                try:
                    result = await session.call_tool(tool_name, tool_args)
                    content = result.content[0].text if result.content else ""
                except Exception as exc:  # noqa: BLE001
                    content = f"Error: {exc}"
                    log(f"  tool {tool_name} failed: {exc}")

            log(f"  -> {content[:200]}")

            if tool_name == "ontology_export" and content:
                # Seed-inclusive exports (step 2 of the workflow, to read the seeds)
                # and empty graphs must not overwrite the persisted ontology.
                if not tool_args.get("include_seeds") and has_ontology_content(content):
                    export_cb(content)
                    log("Ontology exported.")
                else:
                    log("Intermediate ontology export received (skipped persisting seed-inclusive or empty graph).")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": content,
            })
    else:
        log(f"Reached max_iterations ({max_iterations}) — agent stopped.")

    if total_tool_calls == 0:
        log(
            "Warning: the model never called a single tool — it may not support "
            "tool-calling, or the endpoint rejected the `tools` parameter."
        )
