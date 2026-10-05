"""Unit tests for the ontology search agent (cogmaps.ontology.search_agent).

litellm and the OLAF MCP session are faked out — this tests the loop's own
control flow (read-only guard, conversation history, rollback, budget) and
the trace it records, not the LLM.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import cogmaps.ontology.search_agent as search_mod
from cogmaps.ontology.search_agent import SearchConversation, SearchTrace, SearchTurn, ask

_CAR = "http://olaf.local/ontology#Car"
_ENGINE = "http://olaf.local/ontology#Engine"
_CHUNK_ID = "0001b051-4dde-41f0-b713-d37b87391b31"


def _text_response(content: str = ""):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tool_response(name: str, args: dict | None = None, call_id: str = "call-1"):
    tc = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args or {})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeSession:
    """Scripted OLAF MCP session: list_tools + call_tool returning canned text."""

    def __init__(self, tool_results: dict[str, str] | None = None):
        self.tool_results = tool_results or {}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        names = ["concept_search", "chunk_read_batch", "sparql_query", "concept_create", "ontology_summary"]
        return SimpleNamespace(tools=[
            SimpleNamespace(name=n, description="", inputSchema={"type": "object"}) for n in names
        ])

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        text = self.tool_results.get(name, "")
        return SimpleNamespace(content=[SimpleNamespace(text=text)] if text else [])


def _script_completions(monkeypatch, responses: list):
    sent: list[dict] = []
    it = iter(responses)

    def fake_completion(**kwargs):
        sent.append({**kwargs, "messages": list(kwargs["messages"])})
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(search_mod.litellm, "completion", fake_completion)
    return sent


def _ask(session, conversation, question, **kwargs):
    return asyncio.run(ask(session, conversation, question, model="m", api_key="k", **kwargs))


def test_ask_exposes_only_read_only_tools(monkeypatch):
    sent = _script_completions(monkeypatch, [_text_response("answer")])
    _ask(_FakeSession(), SearchConversation("demo"), "q?")
    names = {t["function"]["name"] for t in sent[0]["tools"]}
    assert "concept_create" not in names
    assert {"concept_search", "chunk_read_batch", "sparql_query"} <= names


def test_ask_refuses_a_write_tool_even_if_called(monkeypatch):
    _script_completions(monkeypatch, [_tool_response("concept_create", {"label": "X"}), _text_response("done")])
    session = _FakeSession()
    conversation = SearchConversation("demo")
    _ask(session, conversation, "q?")
    assert session.calls == []
    refusal = next(m for m in conversation.messages if m["role"] == "tool")
    assert "read-only" in refusal["content"]


def test_ask_records_touched_uris_chunks_and_sparql(monkeypatch):
    chunks = [{"id": _CHUNK_ID, "doc_id": "doc.pdf", "chunk_index": 3, "text": "Cars have engines."}]
    session = _FakeSession({
        "concept_search": json.dumps([{"uri": _CAR, "label": "Car", "source_chunk_ids": [_CHUNK_ID]}]),
        "chunk_read_batch": json.dumps(chunks),
        "sparql_query": json.dumps({"form": "SELECT", "rows": [{"o": _ENGINE}]}),
        "ontology_summary": json.dumps({"top": "http://olaf.local/ontology#Everything"}),
    })
    _script_completions(monkeypatch, [
        _tool_response("concept_search", {"query": "car"}),
        _tool_response("ontology_summary"),
        _tool_response("sparql_query", {"query": f"SELECT ?o WHERE {{ <{_CAR}> ?p ?o }}"}),
        _tool_response("chunk_read_batch", {"chunk_ids": [_CHUNK_ID]}),
        _text_response(f"Cars have engines [chunk {_CHUNK_ID}, doc doc.pdf]."),
    ])
    turn = _ask(session, SearchConversation("demo"), "what do cars have?")

    assert _CAR in turn.trace.touched_uris
    assert _ENGINE in turn.trace.touched_uris
    # Whole-store overviews don't count as "touched", nor do chunk URIs.
    assert "http://olaf.local/ontology#Everything" not in turn.trace.touched_uris
    assert not any(u.startswith("urn:olaf:chunk:") for u in turn.trace.touched_uris)
    assert turn.trace.chunks_read[_CHUNK_ID]["doc_id"] == "doc.pdf"
    assert len(turn.trace.sparql_queries) == 1 and turn.trace.sparql_queries[0]["ok"]
    assert turn.cited_chunk_ids == [_CHUNK_ID]


def test_trace_ignores_failed_tool_results():
    trace = SearchTrace()
    trace.record("concept_search", {"query": "x"}, f"Error: no such concept {_CAR}", ok=False)
    assert trace.touched_uris == {}
    assert not trace.tool_calls[0].ok


def test_uris_lose_trailing_punctuation():
    trace = SearchTrace()
    trace.record("concept_get", {}, f"see <{_CAR}>, and ({_ENGINE}).", ok=True)
    assert set(trace.touched_uris) == {_CAR, _ENGINE}


def test_conversation_keeps_history_for_follow_ups(monkeypatch):
    sent = _script_completions(monkeypatch, [_text_response("first"), _text_response("second")])
    conversation = SearchConversation("demo")
    session = _FakeSession()
    _ask(session, conversation, "q1")
    _ask(session, conversation, "and q2?")

    roles_contents = [(m["role"], m.get("content")) for m in sent[1]["messages"]]
    assert ("user", "q1") in roles_contents
    assert ("assistant", "first") in roles_contents
    assert roles_contents[-1] == ("user", "and q2?")
    assert [t.answer for t in conversation.turns] == ["first", "second"]


def test_reset_clears_history_but_keeps_the_system_prompt(monkeypatch):
    _script_completions(monkeypatch, [_text_response("first")])
    conversation = SearchConversation("demo")
    _ask(_FakeSession(), conversation, "q1")
    conversation.reset()
    assert [m["role"] for m in conversation.messages] == ["system"]
    assert "urn:olaf:demo" in conversation.messages[0]["content"]
    assert conversation.turns == []


def test_failed_turn_is_rolled_back(monkeypatch):
    _script_completions(monkeypatch, [_tool_response("concept_search", {"query": "x"}), RuntimeError("LLM down")])
    conversation = SearchConversation("demo")
    with pytest.raises(RuntimeError):
        _ask(_FakeSession(), conversation, "q?")
    assert [m["role"] for m in conversation.messages] == ["system"]
    assert conversation.turns == []


def test_budget_exhausted_forces_a_final_answer_without_tools(monkeypatch):
    sent = _script_completions(monkeypatch, [
        _tool_response("concept_search", {"query": "x"}),
        _tool_response("concept_search", {"query": "y"}),
        _text_response("best effort"),
    ])
    turn = _ask(_FakeSession(), SearchConversation("demo"), "q?", max_iterations=2)
    assert turn.answer == "best effort"
    assert sent[-1]["tool_choice"] == "none"
    assert sent[-1]["messages"][-1]["content"] == search_mod.WRAP_UP_PROMPT


def test_answer_reasoning_block_is_stripped(monkeypatch):
    _script_completions(monkeypatch, [_text_response("<think>hmm</think>The answer.")])
    turn = _ask(_FakeSession(), SearchConversation("demo"), "q?")
    assert turn.answer == "The answer."


def test_cited_chunk_ids_are_deduplicated_in_order():
    a, b = _CHUNK_ID, "11111111-2222-3333-4444-555555555555"
    turn = SearchTurn("q", f"x [chunk {b}, doc d] y [chunk {a}, doc d] z [chunk {b}, doc d]", SearchTrace())
    assert turn.cited_chunk_ids == [b, a]
