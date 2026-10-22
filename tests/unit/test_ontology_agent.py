"""Unit tests for the ontology agent loop (cogmaps.ontology.agent.run_build) and
the runner's final export (cogmaps.ontology.runner.final_export).

litellm and the OLAF MCP session are faked out — this tests the loop's own
control flow (nudging, export persistence guards, fallback), not the LLM.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm

import cogmaps.core.llm_impacts as llm_impacts
import cogmaps.ontology.agent as agent_mod
import cogmaps.ontology.runner as runner_mod
from cogmaps.ontology.agent import run_build
from cogmaps.ontology.runner import final_export

_TTL = "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<http://olaf.local/ontology#Car> a owl:Class ."


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

    def __init__(self, tool_results: dict[str, str] | None = None, raise_on: set[str] | None = None):
        self.tool_results = tool_results or {}
        self.raise_on = raise_on or set()
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return SimpleNamespace(tools=[])

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        if name in self.raise_on:
            raise RuntimeError(f"{name} failed")
        text = self.tool_results.get(name, "")
        if callable(text):
            text = text(args)
        return SimpleNamespace(content=[SimpleNamespace(text=text)] if text else [])


class _PendingChunks:
    """Simulates OLAF chunk status: chunk_list(status=pending) lists chunks until
    chunk_mark_processed has been called for them."""

    def __init__(self, ids):
        self.pending = list(ids)

    def chunk_list(self, args):
        if args.get("status") == "pending":
            return json.dumps([{"id": i, "status": "pending"} for i in self.pending][: args.get("limit", 20)])
        return "[]"

    def chunk_mark_processed(self, args):
        self.pending.remove(args["chunk_id"])
        return "ok"


def _script_completions(monkeypatch, responses: list):
    """Patch litellm.completion to return `responses` in order, recording the messages sent."""
    sent: list[list[dict]] = []
    it = iter(responses)

    def fake_completion(**kwargs):
        sent.append(list(kwargs["messages"]))
        return next(it)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return sent


def _run(session, **kwargs):
    exported: list[str] = []
    logs: list[str] = []
    asyncio.run(run_build(
        session,
        model="m",
        api_key="k",
        doc_filenames=["doc1.pdf"],
        log=logs.append,
        set_progress=lambda cur, tot: None,
        export_cb=exported.append,
        **kwargs,
    ))
    return exported, logs


# ── run_build ────────────────────────────────────────────────────────

def test_run_build_nudges_while_chunks_are_pending_then_finishes(monkeypatch):
    chunks = _PendingChunks(["c1"])
    sent = _script_completions(monkeypatch, [
        _text_response("I think I'm done."),  # c1 still pending -> nudged
        _tool_response("chunk_mark_processed", {"chunk_id": "c1"}),
        _tool_response("ontology_export"),
        _text_response("Finished."),  # nothing pending -> accepted
    ])
    session = _FakeSession({
        "chunk_list": chunks.chunk_list,
        "chunk_mark_processed": chunks.chunk_mark_processed,
        "ontology_export": _TTL,
    })

    exported, logs = _run(session)

    assert len(sent) == 4
    nudge = sent[1][-1]
    assert nudge["role"] == "user"
    assert "Do not stop yet" in nudge["content"]
    assert "doc1.pdf" in nudge["content"]
    assert 'status="pending"' in nudge["content"]
    assert "ontology_orphans" in nudge["content"]
    assert "offset" not in nudge["content"]
    assert exported == [_TTL]
    assert any("no pending chunks remain" in line for line in logs)
    # The pending check is scoped to the requested document
    assert ("chunk_list", {"status": "pending", "limit": 1, "doc_id": "doc1.pdf"}) in session.calls


def test_run_build_accepts_stop_when_nothing_is_pending_even_after_seed_export(monkeypatch):
    # Reading the seeds (include_seeds=true) must not count as "done", and the
    # stop is accepted only because OLAF reports no pending chunk.
    sent = _script_completions(monkeypatch, [
        _tool_response("ontology_export", {"include_seeds": True}),
        _text_response("Done."),
    ])
    session = _FakeSession({"chunk_list": "[]", "ontology_export": _TTL})

    _run(session)

    assert len(sent) == 2


def test_run_build_gives_up_after_max_consecutive_nudges(monkeypatch):
    n = agent_mod.MAX_CONSECUTIVE_NUDGES
    sent = _script_completions(monkeypatch, [_text_response("stop")] * (n + 1))
    session = _FakeSession({"chunk_list": _PendingChunks(["c1"]).chunk_list})

    _, logs = _run(session)

    assert len(sent) == n + 1
    assert any("Re-launch the build to continue" in line and "doc1.pdf" in line for line in logs)


def test_run_build_resets_nudge_count_after_tool_calls(monkeypatch):
    n = agent_mod.MAX_CONSECUTIVE_NUDGES
    responses = [_text_response("stop")] * n + [_tool_response("chunk_read_batch")] + [_text_response("stop")] * (n + 1)
    sent = _script_completions(monkeypatch, responses)
    session = _FakeSession({"chunk_list": _PendingChunks(["c1"]).chunk_list})

    _run(session)

    assert len(sent) == len(responses)


def test_run_build_treats_failed_pending_check_as_unfinished(monkeypatch):
    n = agent_mod.MAX_CONSECUTIVE_NUDGES
    sent = _script_completions(monkeypatch, [_text_response("stop")] * (n + 1))
    session = _FakeSession(raise_on={"chunk_list"})

    _, logs = _run(session)

    assert len(sent) == n + 1
    assert any("unknown (chunk_list failed)" in line for line in logs)


def test_run_build_stops_after_export_without_nudging(monkeypatch):
    sent = _script_completions(monkeypatch, [
        _tool_response("chunk_list"),
        _tool_response("chunk_read_batch"),
        _tool_response("ontology_export"),
        _text_response("Done."),
    ])
    session = _FakeSession({"ontology_export": _TTL})

    exported, _ = _run(session)

    assert len(sent) == 4
    assert not any("Do not stop yet" in str(m.get("content")) for m in sent[-1])
    assert exported == [_TTL]


def test_run_build_skips_persisting_early_seed_export(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("ontology_export", {"include_seeds": True}),
        _text_response("Done."),
    ])
    session = _FakeSession({"ontology_export": _TTL})

    exported, logs = _run(session)

    assert exported == []
    assert any("Intermediate ontology export received" in line for line in logs)


def test_run_build_stops_at_max_iterations_even_if_chunks_are_pending(monkeypatch):
    sent = _script_completions(monkeypatch, [_text_response("stop")] * 2)
    session = _FakeSession({"chunk_list": _PendingChunks(["c1"]).chunk_list})

    exported, logs = _run(session, max_iterations=2)

    assert len(sent) == 2
    assert exported == []
    assert any("Re-launch the build to continue" in line for line in logs)


def test_run_build_sends_and_runs_only_the_build_tools(monkeypatch):
    sent_tools: list[list[str]] = []
    it = iter([_tool_response("sparql_query", {"query": "SELECT * {}"}), _text_response("Done.")])

    def fake_completion(**kwargs):
        sent_tools.append([t["function"]["name"] for t in kwargs["tools"]])
        return next(it)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    class _Session(_FakeSession):
        async def list_tools(self):
            names = ["concept_create", "chunk_read_batch", "sparql_query", "ontology_check", "restriction_add"]
            return SimpleNamespace(tools=[
                SimpleNamespace(name=n, description="", inputSchema={"type": "object"}) for n in names
            ])

    session = _Session({"chunk_list": "[]"})
    _run(session)

    assert sent_tools[0] == ["concept_create", "chunk_read_batch"]
    # A tool outside the list is refused, not run.
    assert ("sparql_query", {"query": "SELECT * {}"}) not in session.calls


def test_run_build_truncates_long_results_but_persists_the_whole_export(monkeypatch):
    big_ttl = _TTL + "\n" + "# padding\n" * agent_mod.MAX_TOOL_RESULT_CHARS
    sent = _script_completions(monkeypatch, [_tool_response("ontology_export"), _text_response("Done.")])
    session = _FakeSession({"chunk_list": "[]", "ontology_export": big_ttl})

    exported, _ = _run(session)

    assert exported == [big_ttl]
    tool_msg = sent[1][-1]
    assert tool_msg["role"] == "tool"
    assert len(tool_msg["content"]) < agent_mod.MAX_TOOL_RESULT_CHARS + 200
    assert "truncated" in tool_msg["content"]


def test_run_build_drops_the_previous_batch_when_reading_the_next(monkeypatch):
    chunks = _PendingChunks(["c1", "c2"])
    sent = _script_completions(monkeypatch, [
        _tool_response("seed_list", call_id="setup"),
        _tool_response("chunk_read_batch", {"chunk_ids": ["c1"]}, call_id="read-1"),
        _tool_response("chunk_mark_processed", {"chunk_id": "c1"}, call_id="mark-1"),
        _tool_response("chunk_read_batch", {"chunk_ids": ["c2"]}, call_id="read-2"),
        _tool_response("chunk_mark_processed", {"chunk_id": "c2"}, call_id="mark-2"),
        _text_response("Done."),
    ])
    session = _FakeSession({
        "seed_list": "[]",
        "chunk_read_batch": lambda args: "chunk text " * 500,
        "chunk_list": chunks.chunk_list,
        "chunk_mark_processed": chunks.chunk_mark_processed,
    })

    _, logs = _run(session)

    # The call after the second batch was read: setup kept, first batch gone.
    after_reset = sent[4]
    ids = [m.get("tool_call_id") for m in after_reset if m["role"] == "tool"]
    assert ids == ["setup", "read-2"]
    note = after_reset[4]
    assert note["role"] == "user"
    assert "1 batch(es) of chunks done so far, 1 chunk(s) marked processed" in note["content"]
    # The batch being worked on is never compacted.
    assert agent_mod._COMPACTED_MARKER not in after_reset[-1]["content"]
    assert any("dropped the previous batch's 4 message(s)" in line for line in logs)
    assert any("no pending chunks remain" in line for line in logs)


def test_build_prompt_asks_to_group_tool_calls():
    from cogmaps.ontology.prompts import SYSTEM_PROMPT

    assert "## Group your tool calls" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("## Group your tool calls") < SYSTEM_PROMPT.index("## Workflow")


# ── compact_history ─────────────────────────────────────────────────

def _history(n_tools: int, size: int) -> list[dict]:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "name": "chunk_read_batch", "content": "x" * size})
    return msgs


def test_compact_history_truncates_only_old_large_tool_results():
    keep = agent_mod.KEEP_RECENT_TOOL_RESULTS
    msgs = _history(keep + 3, agent_mod.COMPACT_THRESHOLD_CHARS + 1)

    assert agent_mod.compact_history(msgs) == 3

    tools = [m for m in msgs if m["role"] == "tool"]
    assert all(agent_mod._COMPACTED_MARKER in m["content"] for m in tools[:3])
    assert all(agent_mod._COMPACTED_MARKER not in m["content"] for m in tools[3:])
    # Structure is preserved: same number of messages, ids still paired
    assert len(msgs) == 2 + 2 * (keep + 3)
    assert [m["tool_call_id"] for m in tools] == [f"t{i}" for i in range(keep + 3)]


def test_compact_history_keeps_small_results_and_is_idempotent():
    keep = agent_mod.KEEP_RECENT_TOOL_RESULTS
    msgs = _history(keep + 2, 200)  # e.g. concept_create URIs
    assert agent_mod.compact_history(msgs) == 0

    big = _history(keep + 1, agent_mod.COMPACT_THRESHOLD_CHARS * 2)
    assert agent_mod.compact_history(big) == 1
    assert agent_mod.compact_history(big) == 0


def test_compact_history_keep_from_compacts_everything_before_the_batch():
    msgs = _history(4, agent_mod.COMPACT_THRESHOLD_CHARS + 1)
    batch_start = 6  # the third assistant turn

    assert agent_mod.compact_history(msgs, keep_from=batch_start) == 2

    tools = [m for m in msgs if m["role"] == "tool"]
    assert [agent_mod._COMPACTED_MARKER in m["content"] for m in tools] == [True, True, False, False]


# ── final_export ─────────────────────────────────────────────────────

def _final_export(session):
    exported: list[str] = []
    logs: list[str] = []
    asyncio.run(final_export(session, "my_onto", export_cb=exported.append, log=logs.append))
    return exported, logs


def test_final_export_uses_session_export(monkeypatch):
    monkeypatch.setattr(runner_mod, "export_oxigraph_ontology_ttl", lambda *a, **k: "unused")
    session = _FakeSession({"ontology_export": _TTL})

    exported, logs = _final_export(session)

    assert exported == [_TTL]
    assert session.calls == [("ontology_export", {"include_seeds": False})]
    assert "Final ontology exported successfully." in logs


def test_final_export_falls_back_to_oxigraph_on_session_error(monkeypatch):
    seen: list[str] = []

    def fake_direct(ontology_id, url=None):
        seen.append(ontology_id)
        return "direct ttl"

    monkeypatch.setattr(runner_mod, "export_oxigraph_ontology_ttl", fake_direct)
    session = _FakeSession(raise_on={"ontology_export"})

    exported, logs = _final_export(session)

    assert exported == ["direct ttl"]
    assert seen == ["my_onto"]
    assert "Final ontology exported via direct Oxigraph connection." in logs


def test_final_export_falls_back_when_session_export_is_empty(monkeypatch):
    monkeypatch.setattr(runner_mod, "export_oxigraph_ontology_ttl", lambda *a, **k: "direct ttl")
    exported, _ = _final_export(_FakeSession({"ontology_export": "# empty"}))
    assert exported == ["direct ttl"]


def test_final_export_writes_nothing_when_both_fail(monkeypatch):
    monkeypatch.setattr(runner_mod, "export_oxigraph_ontology_ttl", lambda *a, **k: None)
    exported, _ = _final_export(_FakeSession(raise_on={"ontology_export"}))
    assert exported == []


def test_run_build_skips_persisting_seed_inclusive_export_late_in_the_run(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("chunk_list"),
        _tool_response("chunk_read_batch"),
        _tool_response("ontology_export", {"include_seeds": True}),
        _tool_response("ontology_export"),
        _text_response("Done."),
    ])
    session = _FakeSession({"ontology_export": _TTL})

    exported, _ = _run(session)

    assert exported == [_TTL]  # only the seed-free export at the end


def test_run_build_skips_persisting_export_without_ontology_terms(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("chunk_list"),
        _tool_response("chunk_read_batch"),
        _tool_response("ontology_export"),
        _text_response("Done."),
    ])
    session = _FakeSession({"ontology_export": "# empty graph\n"})

    exported, _ = _run(session)

    assert exported == []


# ── init_pending_chunks ──────────────────────────────────────────────

def test_init_pending_chunks_makes_fresh_chunks_visible_to_olaf_pending_filter():
    from qdrant_client import QdrantClient, models

    from cogmaps.ontology.runner import OLAF_STATUS_FIELD, init_pending_chunks

    client = QdrantClient(":memory:")
    client.create_collection("col", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    client.upsert("col", [
        models.PointStruct(id=1, vector=[1, 0], payload={"filename": "a.pdf"}),
        models.PointStruct(id=2, vector=[1, 0], payload={"filename": "a.pdf", OLAF_STATUS_FIELD: "processed"}),
        models.PointStruct(id=3, vector=[1, 0], payload={"filename": "b.pdf"}),
    ])
    # Same filter OLAF's chunk_list(status="pending") uses
    pending = models.Filter(must=[models.FieldCondition(key=OLAF_STATUS_FIELD, match=models.MatchValue(value="pending"))])
    assert client.scroll("col", scroll_filter=pending)[0] == []

    init_pending_chunks(client, "col", ["a.pdf"])

    assert [p.id for p in client.scroll("col", scroll_filter=pending)[0]] == [1]
    by_id = {p.id: p.payload for p in client.retrieve("col", [1, 2, 3])}
    assert by_id[2][OLAF_STATUS_FIELD] == "processed"  # already-processed chunk untouched
    assert OLAF_STATUS_FIELD not in by_id[3]  # unselected document untouched


def test_run_build_hides_and_refuses_check_tools(monkeypatch):
    sent: list[dict] = []
    responses = iter([_tool_response("ontology_check"), _text_response("Done.")])

    def fake_completion(**kwargs):
        sent.append({"tools": kwargs["tools"], "messages": list(kwargs["messages"])})
        return next(responses)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    class _ToolsSession(_FakeSession):
        async def list_tools(self):
            names = ["concept_create", "disjoint_add", "ontology_check", "ontology_infer", "entity_delete",
                     "chunk_collection_switch"]
            return SimpleNamespace(tools=[SimpleNamespace(name=n, description="", inputSchema={}) for n in names])

    session = _ToolsSession({"chunk_list": "[]"})
    _run(session)

    assert {t["function"]["name"] for t in sent[0]["tools"]} == {"concept_create", "disjoint_add"}
    assert "ontology_check" not in [name for name, _ in session.calls]
    refusal = sent[1]["messages"][-1]
    assert refusal["role"] == "tool" and "not available" in refusal["content"]


def test_system_prompt_asks_for_disjointness():
    from cogmaps.ontology.prompts import SYSTEM_PROMPT

    assert "**Disjointness** via `disjoint_add`" in SYSTEM_PROMPT


def test_run_build_retries_on_rate_limit_and_transient_errors(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_impacts, "_async_sleep", fake_sleep)

    attempts = iter([
        litellm.RateLimitError("429 quota", "scaleway", "m"),
        litellm.RateLimitError("429 quota", "scaleway", "m"),
        litellm.Timeout("timeout", "m", "scaleway"),
        _text_response("Done."),
    ])
    timeouts: list[float] = []

    def fake_completion(**kwargs):
        timeouts.append(kwargs["timeout"])
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(litellm, "completion", fake_completion)
    session = _FakeSession({"chunk_list": "[]"})

    _, logs = _run(session)

    assert sleeps == [15, 30]
    assert timeouts == [120, 120, 120, 240]
    assert any("Rate limited by the LLM provider — retrying in 15s (1/6)." in line for line in logs)
    assert any("Rate limited by the LLM provider — retrying in 30s (2/6)." in line for line in logs)
    assert any("LLM call timed out — retrying with a 240s timeout (1/1)." in line for line in logs)
    assert any("no pending chunks remain" in line for line in logs)


# ── Seeds ────────────────────────────────────────────────────────────

class _SeedSession(_FakeSession):
    """A session whose store holds the seeds ``loaded`` and that offers the seed tools."""

    def __init__(self, loaded: list[str], **kwargs):
        chunks = _PendingChunks([])
        super().__init__({
            "chunk_list": chunks.chunk_list,
            "seed_list": json.dumps([{"id": s, "graph_uri": f"urn:olaf:seed:{s}"} for s in loaded]),
            "seed_select": lambda args: json.dumps({"selected": args["seed_ids"]}),
            "sparql_query": json.dumps({"rows": [
                {"p": "https://ex.org/fibo/LegalEntity", "g": "urn:olaf:seed:fibo", "label": "legal entity",
                 "children": "4"},
            ]}),
            "concept_search": "[]",
        }, **kwargs)

    async def list_tools(self):
        names = ["concept_search", "seed_reuse", "seed_term_get", "seed_load", "seed_select", "chunk_list"]
        return SimpleNamespace(tools=[SimpleNamespace(name=n, description="", inputSchema={}) for n in names])


def _offered_tools(monkeypatch, responses):
    """Patch litellm.completion like _script_completions, also recording the tools offered."""
    offered: list[set[str]] = []
    sent: list[list[dict]] = []
    it = iter(responses)

    def fake_completion(**kwargs):
        offered.append({t["function"]["name"] for t in kwargs["tools"]})
        sent.append(list(kwargs["messages"]))
        return next(it)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return offered, sent


def test_run_build_selects_the_job_seeds_still_loaded(monkeypatch):
    offered, sent = _offered_tools(monkeypatch, [_text_response("Done.")])
    session = _SeedSession(["fibo", "other"])

    _, logs = _run(session, seed_ids=["fibo", "gone"])

    assert ("seed_select", {"seed_ids": ["fibo"]}) in session.calls
    assert any("gone" in line for line in logs)
    # Seed management is the page's; reusing is the agent's.
    assert offered[0] == {"concept_search", "seed_reuse", "seed_term_get", "chunk_list"}
    system, user = sent[0][0]["content"], sent[0][1]["content"]
    assert "Reuse seed terms" in system
    assert "Seeds selected for this build: fibo." in user
    assert "legal entity <https://ex.org/fibo/LegalEntity> [seed fibo, 4 subclasses]" in user


def test_run_build_forces_include_seeds_on_searches(monkeypatch):
    _offered_tools(monkeypatch, [
        _tool_response("concept_search", {"query": "bank", "include_seeds": False}),
        _text_response("Done."),
    ])
    session = _SeedSession(["fibo"])

    _run(session, seed_ids=["fibo"])

    assert ("concept_search", {"query": "bank", "include_seeds": True}) in session.calls


def test_run_build_without_seeds_selects_none_and_hides_seed_tools(monkeypatch):
    offered, sent = _offered_tools(monkeypatch, [
        _tool_response("concept_search", {"query": "bank"}),
        _text_response("Done."),
    ])
    session = _SeedSession(["fibo"])

    _run(session)

    # Not calling seed_select would leave OLAF's default: every seed in the store.
    assert ("seed_select", {"seed_ids": []}) in session.calls
    assert ("concept_search", {"query": "bank"}) in session.calls
    assert offered[0] == {"concept_search", "chunk_list"}
    assert "Reuse seed terms" not in sent[0][0]["content"]
    assert not any(name == "sparql_query" for name, _ in session.calls)
