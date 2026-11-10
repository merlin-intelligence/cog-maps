"""Unit tests for the ontology editing agent (cogmaps.ontology.editing_agent).

litellm, the OLAF MCP session and Qdrant are faked out — this tests the loop's
own control flow (nothing written while proposing, proposals checked, the user's
choice applied, provenance carried over by relation_replace), not the LLM.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm

from cogmaps.ontology.editing_agent import (
    ChunkSources,
    EditConversation,
    apply_changes,
    ask,
)
from cogmaps.ontology.runner import build_seed_ids, chunk_collections_for

_CAR = "http://olaf.local/ontology#Car"
_VEHICLE = "http://olaf.local/ontology#Vehicle"
_ENGINE = "http://olaf.local/ontology#Engine"
_HAS = "http://olaf.local/ontology#has"
_CONTAINS = "http://olaf.local/ontology#contains"
_CHUNK = "0001b051-4dde-41f0-b713-d37b87391b31"

_SCHEMAS = {
    "concept_search": {"type": "object"},
    "sparql_query": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "chunk_read_batch": {"type": "object"},
    "concept_update": {
        "type": "object",
        "properties": {"uri": {"type": "string"}, "definition": {"type": "string"}},
        "required": ["uri"],
    },
    "concept_merge": {
        "type": "object",
        "properties": {"keep_uri": {"type": "string"}, "merge_uri": {"type": "string"}},
        "required": ["keep_uri", "merge_uri"],
    },
    "relation_add": {"type": "object"},
    "relation_delete": {"type": "object"},
    "relation_sources": {"type": "object"},
    "ontology_create": {"type": "object"},
}


def _text_response(content: str = ""):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])


def _tool_response(name: str, args: dict | None = None, call_id: str = "call-1"):
    tc = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))])


class _FakeSession:
    def __init__(self, tool_results: dict | None = None):
        self.tool_results = tool_results or {}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(name=n, description=f"{n} doc", inputSchema=s) for n, s in _SCHEMAS.items()
        ])

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        result = self.tool_results.get(name, "ok")
        text = result(args) if callable(result) else result
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


class _FakeQdrant:
    def __init__(self, points: dict[str, dict[str, dict]]):
        self.points = points  # collection -> id -> payload

    def retrieve(self, collection, ids, with_payload=True):
        return [SimpleNamespace(id=i, payload=self.points.get(collection, {})[i])
                for i in ids if i in self.points.get(collection, {})]

    def query_points(self, collection_name, query, limit, with_payload=True):
        pts = [SimpleNamespace(id=i, payload=p, score=0.5 + n / 10)
               for n, (i, p) in enumerate(self.points.get(collection_name, {}).items())]
        return SimpleNamespace(points=pts[:limit])


def _sources(points=None):
    return ChunkSources(_FakeQdrant(points or {}), ["col"], lambda _q: [0.0])


def _script_completions(monkeypatch, responses: list):
    sent: list[dict] = []
    it = iter(responses)

    def fake_completion(**kwargs):
        sent.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(it)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return sent


def _ask(session, conversation, request, sources=None):
    return asyncio.run(ask(session, conversation, request, sources=sources or _sources(), model="m", api_key="k"))


def test_tools_are_reads_sources_and_proposals_only(monkeypatch):
    sent = _script_completions(monkeypatch, [_text_response("ok")])
    _ask(_FakeSession(), EditConversation("demo"), "review it")
    names = {t["function"]["name"] for t in sent[0]["tools"]}
    assert {"concept_search", "sparql_query", "source_read", "source_search"} <= names
    assert {"propose_concept_update", "propose_concept_merge", "propose_relation_replace"} <= names
    # No direct write, no OLAF chunk tool (chunks may be in other collections), nothing unrelated.
    assert not names & {"concept_update", "relation_add", "chunk_read_batch", "ontology_create",
                        "propose_ontology_create"}
    propose = next(t for t in sent[0]["tools"] if t["function"]["name"] == "propose_concept_merge")
    assert "reason" in propose["function"]["parameters"]["required"]


def test_a_proposal_writes_nothing(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_concept_merge", {"keep_uri": _VEHICLE, "merge_uri": _CAR, "reason": "same"}),
        _text_response("I propose to merge Car into Vehicle."),
    ])
    session = _FakeSession()
    conversation = EditConversation("demo")
    turn = _ask(session, conversation, "merge car and vehicle")
    assert session.calls == []
    assert [(c.tool, c.args, c.reason, c.status) for c in turn.changes] == [
        ("concept_merge", {"keep_uri": _VEHICLE, "merge_uri": _CAR}, "same", "proposed"),
    ]
    tool_msg = next(m for m in conversation.messages if m["role"] == "tool")
    assert "not applied" in tool_msg["content"]


def test_an_invalid_proposal_is_refused(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_concept_merge", {"keep_uri": _VEHICLE, "reason": "x"}),
        _text_response("sorry"),
    ])
    conversation = EditConversation("demo")
    turn = _ask(_FakeSession(), conversation, "merge")
    assert turn.changes == []
    assert "merge_uri" in next(m for m in conversation.messages if m["role"] == "tool")["content"]


def test_a_direct_write_is_refused(monkeypatch):
    _script_completions(monkeypatch, [_tool_response("concept_update", {"uri": _CAR}), _text_response("ok")])
    session = _FakeSession()
    _ask(session, EditConversation("demo"), "change it")
    assert session.calls == []


def test_replacing_a_relation_that_does_not_exist_is_refused(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_relation_replace", {
            "subject_uri": _CAR, "property_uri": _HAS, "object_value": _ENGINE,
            "new_property_uri": _CONTAINS, "reason": "x",
        }),
        _text_response("ok"),
    ])
    session = _FakeSession({"sparql_query": json.dumps({"boolean": False})})
    turn = _ask(session, EditConversation("demo"), "change it")
    assert turn.changes == []
    assert [name for name, _ in session.calls] == ["sparql_query"]


def test_apply_runs_the_selected_changes_and_discards_the_others(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_concept_update", {"uri": _CAR, "definition": "A car.", "reason": "vague"}),
        _tool_response("propose_concept_merge", {"keep_uri": _VEHICLE, "merge_uri": _CAR, "reason": "same"},
                       call_id="call-2"),
        _text_response("two proposals"),
    ])
    session = _FakeSession()
    conversation = EditConversation("demo")
    turn = _ask(session, conversation, "fix it")
    update, merge = turn.changes
    done = asyncio.run(apply_changes(session, conversation, turn, [update.id]))
    assert done == [update]
    assert (update.status, merge.status) == ("applied", "discarded")
    assert session.calls == [("concept_update", {"uri": _CAR, "definition": "A car."})]
    assert "applied" in conversation.pending_note and "discarded" in conversation.pending_note


def test_a_failed_change_does_not_stop_the_others(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_concept_update", {"uri": _CAR, "definition": "A car.", "reason": "a"}),
        _tool_response("propose_concept_merge", {"keep_uri": _VEHICLE, "merge_uri": _CAR, "reason": "b"},
                       call_id="call-2"),
        _text_response("two"),
    ])
    session = _FakeSession({"concept_update": "Error: Concept not found"})
    conversation = EditConversation("demo")
    turn = _ask(session, conversation, "fix it")
    asyncio.run(apply_changes(session, conversation, turn, [c.id for c in turn.changes]))
    assert [c.status for c in turn.changes] == ["failed", "applied"]
    assert "Concept not found" in turn.changes[0].result


def test_relation_replace_carries_the_source_chunks_over(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_relation_replace", {
            "subject_uri": _CAR, "property_uri": _HAS, "object_value": _ENGINE,
            "new_property_uri": _CONTAINS, "reason": "more precise",
        }),
        _text_response("ok"),
    ])
    session = _FakeSession({
        "sparql_query": json.dumps({"boolean": True}),
        "relation_sources": json.dumps([_CHUNK, "c2"]),
    })
    conversation = EditConversation("demo")
    turn = _ask(session, conversation, "use contains")
    session.calls.clear()
    asyncio.run(apply_changes(session, conversation, turn, [turn.changes[0].id]))
    old = {"subject_uri": _CAR, "property_uri": _HAS, "object_value": _ENGINE}
    new = {**old, "property_uri": _CONTAINS}
    assert session.calls == [
        ("relation_sources", old),
        ("relation_delete", old),
        ("relation_add", {**new, "source_chunk_id": _CHUNK}),
        ("relation_add", {**new, "source_chunk_id": "c2"}),
    ]
    assert turn.changes[0].status == "applied"
    assert "2 source chunk(s)" in turn.changes[0].result


def test_relation_replace_puts_the_old_relation_back_on_failure(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_relation_replace", {
            "subject_uri": _CAR, "property_uri": _HAS, "object_value": _ENGINE,
            "new_object_value": _VEHICLE, "reason": "x",
        }),
        _text_response("ok"),
    ])

    def add(args):
        return "Error: not a term" if args["object_value"] == _VEHICLE else "ok"

    session = _FakeSession({
        "sparql_query": json.dumps({"boolean": True}),
        "relation_sources": json.dumps([_CHUNK]),
        "relation_add": add,
    })
    conversation = EditConversation("demo")
    turn = _ask(session, conversation, "x")
    session.calls.clear()
    asyncio.run(apply_changes(session, conversation, turn, [turn.changes[0].id]))
    old = {"subject_uri": _CAR, "property_uri": _HAS, "object_value": _ENGINE}
    assert session.calls[-1] == ("relation_add", {**old, "source_chunk_id": _CHUNK})
    assert turn.changes[0].status == "failed"


def test_the_agent_is_told_what_happened_to_its_proposals(monkeypatch):
    sent = _script_completions(monkeypatch, [
        _tool_response("propose_concept_update", {"uri": _CAR, "definition": "A car.", "reason": "a"}),
        _text_response("one proposal"),
        _text_response("noted"),
    ])
    session = _FakeSession()
    conversation = EditConversation("demo")
    _ask(session, conversation, "fix it")
    # The user moves on without applying: the proposal is discarded, and the agent told so.
    turn2 = _ask(session, conversation, "never mind")
    assert conversation.turns[0].changes[0].status == "discarded"
    assert "discarded" in sent[-1]["messages"][-1]["content"]
    assert turn2.request == "never mind"
    assert conversation.pending_note is None


def test_source_tools_read_every_chunk_collection():
    sources = ChunkSources(_FakeQdrant({
        "a": {"x": {"text": "from a", "filename": "a.pdf", "chunk_number": 1}},
        "b": {_CHUNK: {"text": "from b", "filename": "b.pdf", "chunk_number": 2}},
    }), ["a", "b"], lambda _q: [0.0])
    read = sources.read([f"urn:olaf:chunk:{_CHUNK}", "missing"])
    assert read[0] == {"id": _CHUNK, "document": "b.pdf", "chunk_number": 2, "text": "from b"}
    assert read[1] == {"id": "missing", "error": "chunk not found"}
    assert [h["id"] for h in sources.search("q", top_k=5)] == ["x", _CHUNK]


class _FakeJobStore:
    def __init__(self, jobs: list[dict]):
        self.jobs = jobs

    def latest_job_for(self, collection, job_type=None, ontology_id=None):
        return next((j for j in self.jobs if j["collection"] == collection and j["job_type"] == job_type
                     and j["ontology_id"] == ontology_id), None)


def test_chunk_collections_and_seeds_follow_copies_and_merges():
    store = _FakeJobStore([
        {"collection": "c1", "ontology_id": "c1__m", "job_type": "merge", "options": {
            "reference": "c1", "other": "c2", "reference_collection": "c1", "other_collection": "c2",
            "seed_ids": ["s1"],
        }},
        {"collection": "c3", "ontology_id": "c3__edit", "job_type": "edit", "options": {
            "source": "c1__m", "source_collection": "c1", "seed_ids": ["s1"],
        }},
    ])
    assert chunk_collections_for(store, "c3", "c3__edit") == ["c1", "c2"]
    assert chunk_collections_for(store, "c4", "c4") == ["c4"]
    assert build_seed_ids(store, "c3", "c3__edit") == ["s1"]


def test_an_empty_reply_is_retried_once(monkeypatch):
    # A reasoning model cut while thinking returns no text and no tool call: asked to answer now.
    sent = _script_completions(monkeypatch, [_text_response(""), _text_response("Here is my review.")])
    turn = _ask(_FakeSession(), EditConversation("demo"), "review")
    assert turn.answer == "Here is my review."
    assert "empty" in sent[1]["messages"][-1]["content"]


def test_two_empty_replies_give_an_explicit_message(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("propose_concept_update", {"uri": _CAR, "definition": "A car.", "reason": "a"}),
        _text_response(""),
        _text_response("<think>…</think>"),
    ])
    conversation = EditConversation("demo")
    turn = _ask(_FakeSession(), conversation, "review")
    assert "produced no answer" in turn.answer and "1 change(s)" in turn.answer
    assert len(turn.changes) == 1
    assert conversation.messages[-1] == {"role": "assistant", "content": turn.answer}


def test_a_long_exploration_without_proposals_is_told_to_propose(monkeypatch):
    import cogmaps.ontology.editing_agent as editing_mod

    monkeypatch.setattr(editing_mod, "EXPLORE_ROUNDS", 2)
    sent = _script_completions(monkeypatch, [
        _tool_response("concept_search", {"query": "a"}),
        _tool_response("concept_search", {"query": "b"}),
        _text_response("done"),
    ])
    _ask(_FakeSession(), EditConversation("demo"), "review")
    assert "explored enough" in sent[2]["messages"][-1]["content"]
    assert not any("explored enough" in (m.get("content") or "") for m in sent[1]["messages"])
