"""Tool-calling agent that answers natural-language questions over an OLAF ontology.

Ported from OLAF's reference search agent
(https://github.com/merlin-intelligence/olaf, ``demos/olaf_searching_agent/agent.py``),
adapted to:
  - keep the conversation in a :class:`SearchConversation` the page stores in
    ``st.session_state``, so follow-up questions work across Streamlit reruns
    even though each question spawns its own OLAF subprocess,
  - call Scaleway through ``litellm``'s native ``scaleway/`` provider the same
    way the building agent does (:mod:`cogmaps.ontology.agent`),
  - record a :class:`SearchTrace` of what the agent looked at (entity URIs,
    chunks read, SPARQL run) so the page can show the part of the ontology and
    the source chunks behind the answer.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession

from cogmaps.config import oxigraph_url, qdrant_api_key
from cogmaps.core.llm_impacts import tracked_completion
from cogmaps.ontology.agent import _compact_history, _summarise, mcp_tools_to_litellm
from cogmaps.ontology.mcp_client import olaf_session
from cogmaps.ontology.olaf_config import build_config_toml
from cogmaps.ontology.search_prompts import WRAP_UP_PROMPT, build_system_prompt
from cogmaps.rag.llm_clients import strip_reasoning

MAX_ITERATIONS = 20
# Tool results longer than this are truncated before being sent to the LLM.
MAX_TOOL_RESULT_CHARS = 20_000

# Tools that only read the store. Anything else (create/update/merge/delete/load/mark…) is
# hidden from the LLM, and refused if it is called anyway.
READ_ONLY_TOOLS = frozenset({
    "chunk_list",
    "chunk_read",
    "chunk_read_batch",
    "concept_list",
    "concept_get",
    "concept_search",
    "concept_semantic_search",
    "property_get",
    "property_search",
    "relation_search",
    "relation_sources",
    "seed_list",
    "ontology_list",
    "ontology_summary",
    "ontology_export",
    "sparql_query",
})

# Tools whose results describe the whole store rather than what the question is
# about — the URIs they return don't count as "touched" by the agent.
_BROAD_TOOLS = frozenset({"ontology_export", "ontology_summary", "ontology_list", "seed_list", "chunk_list"})

_URI_RE = re.compile(r"(?:https?://|urn:olaf:)[^\s\"'<>\\`]+")
# Trailing characters a URI picked out of JSON or prose never ends with.
_URI_TRAILING = ",.;:)]}"
_CHUNK_URI_PREFIX = "urn:olaf:chunk:"
# Citations look like "[chunk <id>, doc <doc_id>]"; ids are cog-maps' Qdrant point UUIDs
# (integers are accepted too, since OLAF itself allows them).
_CITED_CHUNK_RE = re.compile(
    r"\bchunk[\s:#]+([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d+)\b",
    re.IGNORECASE,
)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    preview: str
    ok: bool


@dataclass
class SearchTrace:
    """What the agent looked at while answering one question."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    # Insertion-ordered sets (dict keys) so the page lists things in the order the agent found them.
    touched_uris: dict[str, None] = field(default_factory=dict)
    chunks_read: dict[str, dict] = field(default_factory=dict)
    sparql_queries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, args: dict[str, Any], content: str, ok: bool) -> None:
        self.tool_calls.append(ToolCall(name, args, content[:200], ok))
        if name == "sparql_query":
            self.sparql_queries.append({"query": args.get("query", ""), "ok": ok, "result": content[:500]})
        if not ok:
            return
        if name not in _BROAD_TOOLS:
            sources = [json.dumps(args), content]
            for text in sources:
                for uri in _URI_RE.findall(text):
                    uri = uri.rstrip(_URI_TRAILING)
                    if not uri.startswith(_CHUNK_URI_PREFIX):
                        self.touched_uris.setdefault(uri, None)
        if name in ("chunk_read", "chunk_read_batch"):
            for chunk in _parse_chunks(content):
                self.chunks_read.setdefault(chunk["id"], chunk)


@dataclass
class SearchTurn:
    question: str
    answer: str
    trace: SearchTrace

    @property
    def cited_chunk_ids(self) -> list[str]:
        return list(dict.fromkeys(_CITED_CHUNK_RE.findall(self.answer)))


class SearchConversation:
    """The LLM message history for one ontology, kept between questions for follow-ups."""

    def __init__(self, ontology_id: str):
        self.ontology_id = ontology_id
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(self.ontology_id)},
        ]
        self.turns: list[SearchTurn] = []


def _parse_chunks(content: str) -> list[dict]:
    """Chunks (``{id, doc_id, chunk_index, text}``) from a chunk_read / chunk_read_batch result."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    items = data if isinstance(data, list) else [data]
    return [c for c in items if isinstance(c, dict) and c.get("id") and "text" in c]


async def ask(
    session: ClientSession,
    conversation: SearchConversation,
    question: str,
    *,
    model: str,
    api_key: str,
    log: Callable[[str], None] = lambda _m: None,
    max_iterations: int = MAX_ITERATIONS,
) -> SearchTurn:
    """Answer ``question`` with the read-only OLAF tools, appending to ``conversation``.

    ``log(message)`` is called for every tool call, for live progress on the page.
    """
    tools_result = await session.list_tools()
    tools = mcp_tools_to_litellm([t for t in tools_result.tools if t.name in READ_ONLY_TOOLS])

    messages = conversation.messages
    start = len(messages)
    messages.append({"role": "user", "content": question})
    trace = SearchTrace()
    try:
        answer = await _run_loop(session, messages, tools, trace, model=model,
                                 api_key=api_key, log=log, max_iterations=max_iterations)
    except BaseException:
        # A half-done turn (dangling question, tool calls without results) would
        # break every follow-up — drop it so the conversation stays usable.
        del messages[start:]
        raise

    messages.append({"role": "assistant", "content": answer})
    turn = SearchTurn(question=question, answer=answer, trace=trace)
    conversation.turns.append(turn)
    return turn


async def _run_loop(
    session: ClientSession,
    messages: list[dict[str, Any]],
    tools: list[dict],
    trace: SearchTrace,
    *,
    model: str,
    api_key: str,
    log: Callable[[str], None],
    max_iterations: int,
) -> str:
    async def complete(tool_choice: str):
        return await asyncio.to_thread(
            tracked_completion,
            model=f"scaleway/{model}",
            api_key=api_key,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            # litellm's "scaleway" provider config doesn't whitelist tools/
            # tool_choice as supported params (as of litellm 1.101.0) even
            # though Scaleway's own API supports tool-calling — without this,
            # litellm raises UnsupportedParamsError instead of forwarding them.
            allowed_openai_params=["tools", "tool_choice"],
            max_tokens=4096,
            temperature=0,
            timeout=120,
        )

    for _iteration in range(max_iterations):
        # Earlier questions' chunk texts and SPARQL rows stay in the history — shrink them.
        _compact_history(messages)
        msg = (await complete("auto")).choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            break

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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": await _call_tool(session, tc.function.name, tc.function.arguments, trace, log),
            })
    else:
        # Budget exhausted: force a final answer from what was gathered.
        log(f"Search budget reached ({max_iterations} rounds) — asking for a final answer.")
        messages.append({"role": "user", "content": WRAP_UP_PROMPT})
        answer = (await complete("none")).choices[0].message.content or ""

    return strip_reasoning(answer)


async def _call_tool(
    session: ClientSession, name: str, raw_args: str, trace: SearchTrace, log: Callable[[str], None],
) -> str:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return f"Error: invalid JSON arguments: {raw_args}"

    if name not in READ_ONLY_TOOLS:
        log(f"Refused non read-only tool: {name}")
        return f"Error: tool '{name}' is not available — this agent has read-only access."

    log(f"{name}({_summarise(args)})")
    try:
        result = await session.call_tool(name, args)
        content = result.content[0].text if result.content else ""
        ok = not content.startswith("Error")
    except Exception as exc:  # noqa: BLE001
        content, ok = f"Error: {exc}", False
    if not ok:
        log(f"  {name} failed: {content[:200]}")

    trace.record(name, args, content, ok)
    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            content[:MAX_TOOL_RESULT_CHARS]
            + f"\n… [truncated: {len(content)} chars in total — narrow your query]"
        )
    return content


async def ask_collection(
    conversation: SearchConversation,
    question: str,
    *,
    qdrant_url: str,
    collection: str,
    model: str,
    api_key: str,
    log: Callable[[str], None] = lambda _m: None,
) -> SearchTurn:
    """Spawn an OLAF subprocess for ``collection``'s ontology and answer ``question`` with it.

    Same OLAF config as the builds (:mod:`cogmaps.ontology.runner`), so OLAF's
    startup bootstrap finds the ontology declaration the builds wrote and
    leaves the store untouched — callers must only search ontologies that exist.
    """
    config_dir = tempfile.mkdtemp(prefix="cogmaps_olaf_search_")
    try:
        with open(os.path.join(config_dir, "config.toml"), "w", encoding="utf-8") as f:
            f.write(build_config_toml(
                qdrant_url=qdrant_url,
                qdrant_collection=collection,
                qdrant_api_key=qdrant_api_key(),
                oxigraph_url=oxigraph_url(),
                ontology_id=conversation.ontology_id,
                ontology_name=collection,
            ))
        log("Starting OLAF…")
        async with olaf_session(config_dir) as session:
            return await ask(
                session, conversation, question,
                model=model, api_key=api_key, log=log,
            )
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)
