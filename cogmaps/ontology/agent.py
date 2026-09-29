"""Tool-calling agent loop that drives OLAF's MCP tools to build an ontology.

Ported from OLAF's own reference agent
(https://github.com/merlin-intelligence/olaf, ``demos/olaf_building_agent/agent.py``),
adapted to:
  - stream progress through a ``log()`` callback into
    :class:`cogmaps.ontology.store.OntologyJobStore` instead of stderr logging,
  - scope the initial user turn to the documents selected on the page,
  - call Nebius through ``litellm``'s generic OpenAI-compatible-endpoint
    pattern (``model="openai/<name>"`` + explicit ``api_base``/``api_key``)
    instead of a hardcoded LiteLLM provider,
  - persist the Turtle via an ``export_cb()`` callback instead of a static
    file path from a toml.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import litellm
from mcp import ClientSession

from cogmaps.ontology.prompts import SYSTEM_PROMPT, build_system_prompt

if TYPE_CHECKING:
    from cogmaps.ontology.domain_discovery import DomainBlueprint

MAX_ITERATIONS = 150


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


def _build_user_message(doc_filenames: list[str]) -> str:
    if doc_filenames:
        docs = ", ".join(doc_filenames)
        return (
            f"Build the ontology from the Qdrant chunks of these {len(doc_filenames)} "
            f"document(s) only: {docs}. Scope every chunk_list call to one of these "
            f"document IDs at a time via the doc_id parameter."
        )
    return "Build the ontology from all available Qdrant chunks in the collection."


async def run_build(
    session: ClientSession,
    *,
    model: str,
    api_base: str,
    api_key: str,
    doc_filenames: list[str],
    log: Callable[[str], None],
    set_progress: Callable[[int, int], None],
    export_cb: Callable[[str], None],
    max_iterations: int = MAX_ITERATIONS,
    blueprint: DomainBlueprint | None = None,
    system_prompt: str | None = None,
) -> None:
    """Run the tool-calling loop against an already-initialized OLAF session.

    ``log(message)`` is called for every notable step (tool call + truncated
    result). ``export_cb(turtle)`` is called with the Turtle content whenever
    the model calls ``ontology_export``. ``set_progress(current, total)``
    tracks loop iterations for the page's progress bar.
    """
    tools_result = await session.list_tools()
    tools = mcp_tools_to_litellm(tools_result.tools)
    log(f"Loaded {len(tools)} OLAF tools.")

    active_prompt = system_prompt or (build_system_prompt(blueprint) if blueprint else SYSTEM_PROMPT)
    if blueprint:
        log(f"Configured agent with dynamic domain blueprint: '{blueprint.inferred_domain}' ({len(blueprint.pillars)} pillars)")
    else:
        log("Configured agent with default domain prompt.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": active_prompt},
        {"role": "user", "content": _build_user_message(doc_filenames)},
    ]

    total_tool_calls = 0

    for iteration in range(1, max_iterations + 1):
        log(f"--- Iteration {iteration}/{max_iterations} ---")
        set_progress(iteration, max_iterations)

        response = await asyncio.to_thread(
            litellm.completion,
            model=f"openai/{model}",
            api_base=api_base,
            api_key=api_key,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0,
            timeout=120,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            exported = any(m.get("tool_call_id") and m.get("name") == "ontology_export" for m in messages)
            if not exported and iteration < max_iterations:
                if msg.content:
                    log(f"Agent interim note: {msg.content[:200]}")
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Do not stop yet. Continue processing all remaining document chunks across the selected documents. "
                        "Call chunk_list (using offset or next doc_id) and chunk_read_batch to read chunks in batches, "
                        "extract concepts, properties, and relations, mark chunks processed, and call ontology_export when all chunks are covered."
                    ),
                })
                continue
            log("Agent finished — model returned no further tool calls.")
            if msg.content:
                log(f"Final message: {msg.content}")
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
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            log(f"Tool call: {tool_name}({_summarise(tool_args)})")
            total_tool_calls += 1

            try:
                result = await session.call_tool(tool_name, tool_args)
                content = result.content[0].text if result.content else ""
            except Exception as exc:  # noqa: BLE001
                content = f"Error: {exc}"
                log(f"  tool {tool_name} failed: {exc}")

            log(f"  -> {content[:200]}")

            if tool_name == "ontology_export" and content:
                # Avoid overwriting the persisted file with global seeds or empty models during early exploration
                if iteration > 2 and ("owl:Class" in content or "a <http://www.w3.org/2002/07/owl#Class>" in content):
                    export_cb(content)
                    log("Ontology exported.")
                else:
                    log("Intermediate ontology export received (skipped persisting seed-only or pre-build graph).")

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
