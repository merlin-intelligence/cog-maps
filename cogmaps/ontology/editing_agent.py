"""Conversational agent that reviews and modifies an ontology: it proposes, the user validates.

An edited ontology is a copy of another one (``ontology_import``, provenance and
semantic index included — see :meth:`cogmaps.ontology.runner.OntologyJobRunner`),
which this agent then modifies through a chat on the building page:

  - Each user message runs a tool-calling loop over OLAF's read tools, plus
    ``source_read`` / ``source_search`` on the source chunks (read straight from
    Qdrant: a copied or merged ontology's chunks can live in other collections
    than the one OLAF is configured with).
  - The write tools are offered as ``propose_<tool>``: a call is checked against
    OLAF's schema and recorded as a :class:`ProposedChange`, nothing is written.
  - The page lists the proposals; the user picks those to apply, and
    :func:`apply_changes` runs them through OLAF's own write tools — which keep
    provenance (``urn:olaf:extractedFrom``) and the semantic index in step.
    OLAF has no tool to change a relation in place, so ``relation_replace`` is
    done here: its source chunks are read, the relation deleted, the new one
    added with the same chunks (and the old one put back if that fails).

The conversation (:class:`EditConversation`) lives in the page's session state,
so follow-ups work across Streamlit reruns although each message spawns its own
OLAF subprocess, as with the search agent (:mod:`cogmaps.ontology.search_agent`).
"""
from __future__ import annotations

import contextlib
import copy
import json
import re
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from string import Template
from typing import Any

import jsonschema
from mcp import ClientSession
from qdrant_client import QdrantClient

from cogmaps.config import chat_max_tokens, qdrant_api_key
from cogmaps.core.llm_impacts import INTERACTIVE_RETRY, acompletion_with_retry
from cogmaps.ontology.agent import (
    compact_history,
    mcp_tools_to_litellm,
    select_seeds,
    summarise_args,
)
from cogmaps.ontology.editing_prompts import (
    APPLIED_NOTE,
    DISCARDED_NOTE,
    EMPTY_REPLY_PROMPT,
    EXPLORED_ENOUGH_PROMPT,
    WRAP_UP_PROMPT,
    build_system_prompt,
)
from cogmaps.ontology.mcp_client import olaf_session
from cogmaps.ontology.olaf_cli import run_olaf, write_config
from cogmaps.ontology.olaf_config import ontology_display_name, restore_oxigraph_graph
from cogmaps.ontology.search_agent import READ_ONLY_TOOLS
from cogmaps.ontology.seeds import ensure_ontology_index
from cogmaps.rag.llm_clients import strip_reasoning

MAX_ITERATIONS = 20
# Without a proposal by then, the agent is told to stop exploring: a broad review otherwise reads
# the whole ontology turn after turn (minutes) before proposing anything.
EXPLORE_ROUNDS = 10
MAX_TOOL_RESULT_CHARS = 20_000
MAX_CHUNK_CHARS = 3000         # each chunk's text returned by source_read / source_search is cut beyond this
SOURCE_SEARCH_TOP_K = 5
REINDEX_TIMEOUT = 600

# OLAF's chunk tools only read the collection OLAF is configured with: hidden, in favour of
# source_read / source_search, which read every collection the ontology's chunks come from.
READ_TOOLS = READ_ONLY_TOOLS - {"chunk_list", "chunk_read", "chunk_read_batch"}

# OLAF write tools the agent may propose (as propose_<name>). Nothing else writes.
PROPOSABLE_TOOLS = frozenset({
    "concept_create", "concept_update", "concept_merge", "individual_create",
    "property_create", "property_update",
    "relation_add", "relation_delete",
    "restriction_add", "restriction_delete",
    "entity_delete", "disjoint_add", "seed_reuse",
})
RELATION_REPLACE = "relation_replace"
PROPOSE_PREFIX = "propose_"

_RELATION_PROPS = {
    "subject_uri": {"type": "string", "description": "Subject URI"},
    "property_uri": {"type": "string", "description": "Property URI"},
    "object_value": {"type": "string", "description": "Object URI, or the literal value"},
    "is_literal": {"type": "boolean", "description": "Whether the object is a literal (default: false)"},
    "datatype": {"type": "string", "description": "Literal datatype URI, if any"},
}
RELATION_REPLACE_SCHEMA = {
    "type": "object",
    "properties": {
        **_RELATION_PROPS,
        **{f"new_{k}": {**v, "description": f"New {v['description'][0].lower()}{v['description'][1:]} "
                                            "(default: unchanged)"}
           for k, v in _RELATION_PROPS.items()},
    },
    "required": ["subject_uri", "property_uri", "object_value"],
}
RELATION_REPLACE_DESCRIPTION = (
    "Change a relation (subject, property and/or object) while keeping its provenance: the source chunks of "
    "the current relation are carried over to the new one. Give the current relation, then only the new_* "
    "fields that change."
)

SOURCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "source_read",
            "description": "Read source chunks by id (the ids of urn:olaf:chunk:{id}, or from source_search / "
                           "relation_sources / concept_get). Returns id, document, chunk number and text.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "source_search",
            "description": "Semantic search over the source chunks of the ontology's documents: the passages "
                           "closest in meaning to the query. Use it to find the chunk that supports a new "
                           "concept or relation (its source_chunk_id).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": f"Default {SOURCE_SEARCH_TOP_K}, max 10"},
                },
                "required": ["query"],
            },
        },
    },
]

_URI_RE = re.compile(r"^(?:https?://|urn:)")


def _local(value: Any) -> str:
    s = str(value)
    return re.split(r"[#/]", s.rstrip("#/"))[-1] if _URI_RE.match(s) else s


# ── Proposals ────────────────────────────────────────────────────────────

@dataclass
class ProposedChange:
    id: int
    tool: str           # an OLAF write tool, or "relation_replace"
    args: dict[str, Any]
    reason: str = ""
    status: str = "proposed"  # proposed | applied | failed | discarded
    result: str = ""

    def describe(self) -> str:
        """One line for the page: the change, with URIs shortened to their local names."""
        if self.tool == RELATION_REPLACE:
            old = [self.args.get(k) for k in ("subject_uri", "property_uri", "object_value")]
            new = [self.args.get(f"new_{k}") or v for k, v in zip(("subject_uri", "property_uri", "object_value"), old,
                                                                   strict=True)]
            return f"relation: {' '.join(map(_local, old))}  →  {' '.join(map(_local, new))}"
        if self.tool in ("relation_add", "relation_delete"):
            triple = " ".join(_local(self.args.get(k, "")) for k in ("subject_uri", "property_uri", "object_value"))
            return f"{self.tool.replace('_', ' ')}: {triple}"
        details = ", ".join(
            f"{k}={_local(v) if isinstance(v, str) and len(v) < 120 else str(v)[:120] + '…'}"
            for k, v in self.args.items() if k not in ("language",)
        )
        return f"{self.tool.replace('_', ' ')}: {details}"

    def as_note(self) -> str:
        return f"- #{self.id} {self.describe()} → {self.status}" + (f": {self.result}" if self.result else "")


@dataclass
class EditTurn:
    request: str
    answer: str
    changes: list[ProposedChange] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    backup_path: str | None = None  # the ontology before this turn's changes were applied

    @property
    def pending(self) -> list[ProposedChange]:
        return [c for c in self.changes if c.status == "proposed"]


class EditConversation:
    """The LLM message history for one edited ontology, kept between messages."""

    def __init__(self, ontology_id: str):
        self.ontology_id = ontology_id
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(self.ontology_id)},
        ]
        self.turns: list[EditTurn] = []
        # What happened to the last proposals, told to the agent with the next message.
        self.pending_note: str | None = None
        self._next_id = 1

    def new_change(self, tool: str, args: dict, reason: str) -> ProposedChange:
        change = ProposedChange(self._next_id, tool, args, reason)
        self._next_id += 1
        return change

    def settle(self) -> None:
        """Proposals still pending when the user moves on are discarded — and the agent told so."""
        if self.turns and (pending := self.turns[-1].pending):
            for c in pending:
                c.status = "discarded"
            if self.pending_note is None:
                self.pending_note = DISCARDED_NOTE


# ── Source chunks (read from Qdrant directly) ────────────────────────────

class ChunkSources:
    """The ontology's source chunks, in the Qdrant collections they come from."""

    def __init__(self, client: QdrantClient, collections: Sequence[str], embed_query: Callable[[str], Any]):
        self.client = client
        self.collections = list(dict.fromkeys(collections))
        self.embed_query = embed_query

    @staticmethod
    def _chunk(point) -> dict:
        payload = point.payload or {}
        text = payload.get("text", "")
        return {
            "id": str(point.id),
            "document": payload.get("filename", "unknown"),
            "chunk_number": payload.get("chunk_number"),
            "text": text[:MAX_CHUNK_CHARS] + (" …" if len(text) > MAX_CHUNK_CHARS else ""),
        }

    def read(self, chunk_ids: list[str]) -> list[dict]:
        found: dict[str, dict] = {}
        missing = [str(c).removeprefix("urn:olaf:chunk:") for c in chunk_ids]
        for collection in self.collections:
            if not missing:
                break
            try:
                points = self.client.retrieve(collection, ids=missing, with_payload=True)
            except Exception:  # noqa: BLE001 — invalid id format for this collection, or it is gone
                continue
            for p in points:
                found[str(p.id)] = self._chunk(p)
            missing = [c for c in missing if c not in found]
        return [*found.values(), *({"id": c, "error": "chunk not found"} for c in missing)]

    def search(self, query: str, top_k: int = SOURCE_SEARCH_TOP_K) -> list[dict]:
        vector = self.embed_query(query)
        vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        hits = []
        for collection in self.collections:
            try:
                res = self.client.query_points(collection_name=collection, query=vector, limit=top_k,
                                               with_payload=True)
            except Exception:  # noqa: BLE001
                continue
            hits += [(p.score, {**self._chunk(p), "score": round(p.score, 3)}) for p in res.points]
        hits.sort(key=lambda h: h[0], reverse=True)
        return [h for _, h in hits[:top_k]]


# ── The conversation loop ─────────────────────────────────────────────────

def proposal_tools(mcp_tools: list) -> tuple[list[dict], dict[str, dict]]:
    """``propose_*`` tool definitions for the LLM, and the schema each change is checked against
    (keyed by the OLAF tool name, or ``relation_replace``)."""
    schemas: dict[str, dict] = {}
    descriptions: dict[str, str] = {}
    for t in mcp_tools:
        if t.name in PROPOSABLE_TOOLS:
            schemas[t.name] = t.inputSchema or {"type": "object"}
            descriptions[t.name] = t.description or ""
    if {"relation_add", "relation_delete", "relation_sources"} <= {t.name for t in mcp_tools}:
        schemas[RELATION_REPLACE] = RELATION_REPLACE_SCHEMA
        descriptions[RELATION_REPLACE] = RELATION_REPLACE_DESCRIPTION
    tools = []
    for name, schema in schemas.items():
        params = copy.deepcopy(schema)
        params.setdefault("properties", {})["reason"] = {
            "type": "string", "description": "Why this change, in one sentence (shown to the user)",
        }
        params["required"] = [*params.get("required", []), "reason"]
        tools.append({"type": "function", "function": {
            "name": PROPOSE_PREFIX + name,
            "description": "PROPOSE this change — recorded only, applied once the user validates it. "
                           + descriptions[name],
            "parameters": params,
        }})
    return tools, schemas


async def ask(
    session: ClientSession,
    conversation: EditConversation,
    request: str,
    *,
    sources: ChunkSources,
    model: str,
    api_key: str,
    log: Callable[[str], None] = lambda _m: None,
    max_iterations: int = MAX_ITERATIONS,
) -> EditTurn:
    """Answer ``request``: read the ontology and the chunks, record the changes proposed."""
    mcp_tools = (await session.list_tools()).tools
    propose, schemas = proposal_tools(mcp_tools)
    tools = [*mcp_tools_to_litellm([t for t in mcp_tools if t.name in READ_TOOLS]), *SOURCE_TOOLS, *propose]

    conversation.settle()
    content = request
    if conversation.pending_note:
        content = f"{conversation.pending_note}\n\n{request}"
    messages = conversation.messages
    start = len(messages)
    messages.append({"role": "user", "content": content})
    turn = EditTurn(request=request, answer="")
    try:
        turn.answer = await _run_loop(session, conversation, turn, tools, schemas, sources,
                                      model=model, api_key=api_key, log=log, max_iterations=max_iterations)
    except BaseException:
        # A half-done turn (dangling message, tool calls without results) would break every follow-up.
        del messages[start:]
        raise
    if not turn.answer:
        turn.answer = empty_answer_message(model, len(turn.changes))
    messages.append({"role": "assistant", "content": turn.answer})
    conversation.pending_note = None
    conversation.turns.append(turn)
    return turn


async def _run_loop(
    session: ClientSession,
    conversation: EditConversation,
    turn: EditTurn,
    tools: list[dict],
    schemas: dict[str, dict],
    sources: ChunkSources,
    *,
    model: str,
    api_key: str,
    log: Callable[[str], None],
    max_iterations: int,
) -> str:
    messages = conversation.messages

    async def complete(tool_choice: str):
        return await acompletion_with_retry(
            retry=INTERACTIVE_RETRY,
            log=log,
            model=f"scaleway/{model}",
            api_key=api_key,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            # Same litellm workaround as the other agents: the "scaleway" provider doesn't
            # whitelist tools/tool_choice.
            allowed_openai_params=["tools", "tool_choice"],
            # Reasoning models think before they answer: a broad review can spend 4096 tokens
            # thinking and be cut before writing anything. Same budget as the /ask/ chat.
            max_tokens=chat_max_tokens(),
            temperature=0,
            timeout=180,
        )

    nudged = False
    for iteration in range(max_iterations):
        if iteration == EXPLORE_ROUNDS and not turn.changes:
            messages.append({"role": "user", "content": EXPLORED_ENOUGH_PROMPT})
        compact_history(messages)
        choice = (await complete("auto")).choices[0]
        msg = choice.message
        if not msg.tool_calls:
            if text := strip_reasoning(msg.content or "").strip():
                return text
            # Nothing said: cut while reasoning (finish_reason "length"), or an empty reply.
            # Asked once to answer now; a second empty reply ends the turn (see ask()).
            log(f"Empty reply from the model (finish_reason={getattr(choice, 'finish_reason', None)}).")
            if nudged:
                return ""
            nudged = True
            messages.append({"role": "user", "content": EMPTY_REPLY_PROMPT})
            continue
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            result = await _call_tool(session, conversation, turn, schemas, sources,
                                      tc.function.name, tc.function.arguments, log)
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + f"\n… [truncated: {len(result)} chars — narrow your query]"
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})

    log(f"Tool budget reached ({max_iterations} rounds) — asking for a final answer.")
    messages.append({"role": "user", "content": WRAP_UP_PROMPT})
    return strip_reasoning((await complete("none")).choices[0].message.content or "")


async def _call_tool(
    session: ClientSession,
    conversation: EditConversation,
    turn: EditTurn,
    schemas: dict[str, dict],
    sources: ChunkSources,
    name: str,
    raw_args: str,
    log: Callable[[str], None],
) -> str:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return f"Error: invalid JSON arguments: {raw_args}"
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object."
    turn.tool_log.append(f"{name}({summarise_args(args)})")
    log(f"{name}({summarise_args(args)})")

    if name.startswith(PROPOSE_PREFIX):
        return await _propose(session, conversation, turn, schemas, name.removeprefix(PROPOSE_PREFIX), args)
    if name == "source_read":
        return json.dumps(sources.read(list(args.get("chunk_ids") or [])), ensure_ascii=False)
    if name == "source_search":
        top_k = max(1, min(int(args.get("top_k") or SOURCE_SEARCH_TOP_K), 10))
        return json.dumps(sources.search(str(args.get("query", "")), top_k), ensure_ascii=False)
    if name not in READ_TOOLS:
        log(f"  refused: {name}")
        return f"Error: tool '{name}' is not available. To change the ontology, use the propose_* tools."
    try:
        result = await session.call_tool(name, args)
        content = result.content[0].text if result.content else ""
    except Exception as exc:  # noqa: BLE001
        content = f"Error: {exc}"
    if content.startswith("Error"):
        log(f"  {name} failed: {content[:200]}")
    return content


async def _propose(
    session: ClientSession,
    conversation: EditConversation,
    turn: EditTurn,
    schemas: dict[str, dict],
    tool: str,
    args: dict,
) -> str:
    if tool not in schemas:
        return f"Error: '{tool}' cannot be proposed."
    reason = str(args.pop("reason", "") or "")
    try:
        jsonschema.validate(args, schemas[tool])
    except jsonschema.ValidationError as e:
        return f"Error: invalid proposal: {e.message}"
    if tool in ("relation_delete", RELATION_REPLACE) and not await _relation_exists(session, conversation, args):
        return ("Error: this relation is not in the ontology (check the URIs with relation_search). "
                "Nothing was proposed.")
    change = conversation.new_change(tool, args, reason)
    turn.changes.append(change)
    return f"Proposed as change #{change.id} — not applied: the user will review it."


async def _relation_exists(session: ClientSession, conversation: EditConversation, args: dict) -> bool:
    """Whether the relation a proposal deletes or replaces is there. Literals are not checked
    (their exact lexical form and datatype are easy to get slightly wrong): OLAF reports it then."""
    if args.get("is_literal"):
        return True
    s, p, o = (args.get(k, "") for k in ("subject_uri", "property_uri", "object_value"))
    if not all(_URI_RE.match(str(x)) for x in (s, p, o)) or any(c in f"{s}{p}{o}" for c in "<> "):
        return False
    query = f"ASK {{ GRAPH <urn:olaf:{conversation.ontology_id}> {{ <{s}> <{p}> <{o}> }} }}"
    try:
        result = await session.call_tool("sparql_query", {"query": query})
        return bool(json.loads(result.content[0].text).get("boolean"))
    except Exception:  # noqa: BLE001 — can't tell: let the apply step report it
        return True


def empty_answer_message(model: str, proposals: int) -> str:
    """What the page shows when the model produced no answer at all."""
    return (
        f"⚠️ {model} produced no answer"
        + (f" (it proposed {proposals} change(s), listed below)" if proposals else "")
        + " — it most likely used up its token budget reasoning. Try a narrower request (one part of the "
          "ontology, one kind of problem), raise CHAT_MAX_TOKENS, or pick another model."
    )


# ── Applying the validated changes ────────────────────────────────────────

async def _write(session: ClientSession, tool: str, args: dict) -> str:
    result = await session.call_tool(tool, args)
    content = result.content[0].text if result.content else ""
    if content.startswith("Error"):
        raise RuntimeError(content.removeprefix("Error:").strip())
    return content


def _relation_args(args: dict, prefix: str = "", fallback: dict | None = None) -> dict:
    fallback = fallback or {}
    out = {}
    for k in _RELATION_PROPS:
        v = args.get(prefix + k)
        if v is None or v == "":
            v = fallback.get(k)
        if v is not None:
            out[k] = v
    return out


async def _replace_relation(session: ClientSession, args: dict) -> str:
    old = _relation_args(args)
    new = _relation_args(args, "new_", old)
    if new == old:
        raise RuntimeError("the new relation is the same as the current one")
    chunks = json.loads(await _write(session, "relation_sources", old))
    await _write(session, "relation_delete", old)
    try:
        for chunk in chunks or [None]:
            await _write(session, "relation_add", {**new, **({"source_chunk_id": chunk} if chunk else {})})
    except Exception:
        # Undo a partial add, and put the old relation back with its provenance, before reporting.
        with contextlib.suppress(RuntimeError):
            await _write(session, "relation_delete", new)
        for chunk in chunks or [None]:
            await _write(session, "relation_add", {**old, **({"source_chunk_id": chunk} if chunk else {})})
        raise
    return f"ok — {len(chunks)} source chunk(s) carried over" if chunks else "ok — the relation had no source chunk"


async def apply_changes(
    session: ClientSession,
    conversation: EditConversation,
    turn: EditTurn,
    selected_ids: Sequence[int],
    log: Callable[[str], None] = lambda _m: None,
) -> list[ProposedChange]:
    """Apply the turn's selected proposals in order (the others are discarded). Each one is applied
    or fails on its own. The results are told to the agent with the next message."""
    selected = set(selected_ids)
    for change in turn.pending:
        if change.id not in selected:
            change.status = "discarded"
            continue
        try:
            if change.tool == RELATION_REPLACE:
                change.result = await _replace_relation(session, change.args)
            else:
                change.result = (await _write(session, change.tool, change.args))[:300]
            change.status = "applied"
        except Exception as exc:  # noqa: BLE001
            change.status, change.result = "failed", str(exc)[:300]
        log(change.as_note())
    done = [c for c in turn.changes if c.status in ("applied", "failed")]
    conversation.pending_note = (
        Template(APPLIED_NOTE).substitute(results="\n".join(c.as_note() for c in turn.changes))
        if done else DISCARDED_NOTE
    )
    return done


# ── OLAF sessions for the page ────────────────────────────────────────────

@asynccontextmanager
async def olaf_for(
    *, qdrant_url: str, collection: str, ontology_id: str, seed_ids: Sequence[str], log: Callable[[str], None],
) -> AsyncIterator[ClientSession]:
    """An OLAF session on the ontology, with the seeds it was built with selected."""
    config_dir = tempfile.mkdtemp(prefix="cogmaps_olaf_edit_")
    try:
        write_config(config_dir, qdrant_url=qdrant_url, qdrant_collection=collection, ontology_id=ontology_id,
                     ontology_name=ontology_display_name(collection, ontology_id))
        ensure_ontology_index(ontology_id, config_dir, QdrantClient(url=qdrant_url, api_key=qdrant_api_key()), log)
        log("Starting OLAF…")
        async with olaf_session(config_dir) as session:
            await select_seeds(session, list(seed_ids), log)
            yield session
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)


def restore_and_reindex(*, qdrant_url: str, collection: str, ontology_id: str, ttl: str) -> None:
    """Put the ontology back to ``ttl`` — its graph, and its semantic index rebuilt from it."""
    restore_oxigraph_graph(ontology_id, ttl)
    config_dir = tempfile.mkdtemp(prefix="cogmaps_olaf_undo_")
    try:
        write_config(config_dir, qdrant_url=qdrant_url, qdrant_collection=collection, ontology_id=ontology_id,
                     ontology_name=ontology_display_name(collection, ontology_id))
        run_olaf(["reindex", ontology_id], config_dir, timeout=REINDEX_TIMEOUT)
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)


__all__ = [
    "ChunkSources", "EditConversation", "EditTurn", "ProposedChange",
    "apply_changes", "ask", "olaf_for", "proposal_tools", "restore_and_reindex",
]
