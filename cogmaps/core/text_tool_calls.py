"""Recover tool calls a model wrote as text instead of structured ``tool_calls``.

Models emit tool calls as text tokens, and the inference server (vLLM on Scaleway)
parses them into ``tool_calls`` with a parser specific to each model. When the model
gets its own syntax slightly wrong, that parser fails and the call is left as plain
text in ``content`` — seen with GLM 5.3, which sometimes closes with ``<tool_call>``
instead of ``</tool_call>``:

    <tool_call>
    {"name": "chunk_mark_processed", "arguments": {"chunk_id": "d342…"}}
    <tool_call>

The agent loops would then read "no tool call" as the model stopping. This module
recognises the two ``<tool_call>`` formats (both model-agnostic: Qwen uses the JSON
one, GLM both) and turns them back into tool calls:

- JSON (Hermes): ``<tool_call>{"name": …, "arguments": {…}}</tool_call>``
- GLM native:    ``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``

The closing tag is optional (or may be a repeated ``<tool_call>``).
"""
from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from typing import Any

# A block runs from <tool_call> to the next </tool_call>, <tool_call> or the end.
_BLOCK_RE = re.compile(r"<tool_call>(.*?)(?=</tool_call>|<tool_call>|\Z)", re.DOTALL)
_TAG_RE = re.compile(r"</?tool_call>")
_GLM_NAME_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*(?=<arg_key>|$)", re.DOTALL)
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def _parse_json_block(body: str) -> tuple[str, dict] | None:
    body = body.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    args = data.get("arguments", data.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return (data["name"], args) if isinstance(args, dict) else None


def _parse_glm_block(body: str) -> tuple[str, dict] | None:
    m = _GLM_NAME_RE.match(body)
    if not m:
        return None
    args: dict[str, Any] = {}
    for key, raw in _GLM_ARG_RE.findall(body):
        raw = raw.strip()
        try:
            args[key.strip()] = json.loads(raw)  # numbers, booleans, lists…
        except json.JSONDecodeError:
            args[key.strip()] = raw
    return m.group(1), args


def parse_text_tool_calls(content: str | None) -> list[tuple[str, dict]]:
    """``(name, arguments)`` of every tool call written as text in ``content``.

    Empty if any ``<tool_call>`` block can't be parsed — better to leave the
    message alone than run only part of what the model asked for.
    """
    if not content or "<tool_call>" not in content:
        return []
    calls = []
    for body in _BLOCK_RE.findall(content):
        if not body.strip():
            continue  # the space between a block and its (wrong) closing <tool_call>
        call = _parse_json_block(body) or _parse_glm_block(body)
        if call is None:
            return []
        calls.append(call)
    return calls


def recover_text_tool_calls(msg: Any) -> int:
    """If ``msg`` has no ``tool_calls`` but its content holds some, move them there.

    ``msg`` (a litellm message) is modified in place: ``tool_calls`` gets one entry
    per recovered call, with a generated id, and ``content`` keeps only the text
    outside the ``<tool_call>`` blocks. Returns how many calls were recovered.
    """
    if getattr(msg, "tool_calls", None):
        return 0
    calls = parse_text_tool_calls(getattr(msg, "content", None))
    if not calls:
        return 0
    msg.tool_calls = [
        SimpleNamespace(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
        )
        for name, args in calls
    ]
    first = msg.content.index("<tool_call>")
    msg.content = msg.content[:first].strip() or None
    return len(calls)
