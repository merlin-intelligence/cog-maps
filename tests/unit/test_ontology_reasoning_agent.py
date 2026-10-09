"""Unit tests for the ontology reasoning agent (cogmaps.ontology.reasoning_agent).

litellm and the OLAF MCP session are faked out — this tests the agent's own
control flow (dedup → repair → enrich → infer, stop conditions, backup/export,
tool filtering), not the LLM or the reasoner.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm

from cogmaps.ontology.reasoning_agent import FIX_TOOLS, prune_history, report_counts, run_reasoning

_TTL = "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<http://olaf.local/ontology#Car> a owl:Class ."
_NS = "http://olaf.local/ontology#"
_CLEAN = {"reasoner": {"consistent": True, "unsatisfiable_classes": []}, "integrity": {}}
_ORPHAN = {"uri": _NS + "Wheel", "label": "Wheel", "kind": "class"}
_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_DISJOINT = "http://www.w3.org/2002/07/owl#disjointWith"

# Steps off, so a test only exercises what it turns on.
_REPAIR_ONLY = {"dedup": False, "enrich_disjointness": False, "infer": False}


def _text_response(content: str = ""):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tool_response(name: str, args: dict | None = None, call_id: str = "call-1"):
    tc = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeSession:
    """Scripted OLAF session. ``checks`` are the successive (ontology_check report, orphans)
    pairs, the last one repeating; ``handlers`` override any tool's result."""

    def __init__(self, checks: list[tuple[dict, list]], handlers: dict | None = None):
        self.checks = list(checks)
        self.handlers = handlers or {}
        self.tools = sorted(FIX_TOOLS | {"chunk_mark_processed", "ontology_export"})
        self.calls: list[tuple[str, dict]] = []
        self._orphans: list = []

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=n, description="", inputSchema={}) for n in self.tools])

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        if name in self.handlers:
            text = self.handlers[name](args)
        elif name == "ontology_check":
            report, self._orphans = self.checks.pop(0) if len(self.checks) > 1 else self.checks[0]
            text = json.dumps(report)
        elif name == "ontology_orphans":
            text = json.dumps(self._orphans)
        elif name == "ontology_export":
            text = _TTL
        elif name == "concept_get":
            text = json.dumps({"uri": args["uri"], "label": "Wheel", "triples": [], "source_chunk_ids": []})
        elif name == "sparql_query":
            text = json.dumps({"rows": []})
        else:
            text = "ok"
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _script_completions(monkeypatch, responses: list):
    sent: list[dict] = []
    it = iter(responses)

    def fake_completion(**kwargs):
        sent.append(kwargs)
        return next(it)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return sent


def _run(session, **kwargs):
    out = {"backups": [], "exports": [], "logs": []}
    summary = asyncio.run(run_reasoning(
        session,
        model="m",
        api_key="k",
        ontology_id="main",
        log=out["logs"].append,
        set_progress=lambda cur, tot: None,
        backup_cb=out["backups"].append,
        export_cb=out["exports"].append,
        **kwargs,
    ))
    return summary, out


def _labels(summary: dict) -> list[str]:
    return [c["label"] for c in summary["checks"]]


# ── Repair ───────────────────────────────────────────────────────────

def test_clean_ontology_stops_without_llm_backup_or_export(monkeypatch):
    sent = _script_completions(monkeypatch, [])
    summary, out = _run(_FakeSession([(_CLEAN, [])]), **_REPAIR_ONLY)

    assert sent == []
    assert _labels(summary) == ["Round 1"]
    assert summary["checks"][0]["consistent"] is True
    assert summary["changes"] == 0
    assert out["backups"] == [] and out["exports"] == []
    assert any("Nothing left to fix" in line for line in out["logs"])


def test_fixes_orphan_backs_up_first_then_exports(monkeypatch):
    sent = _script_completions(monkeypatch, [
        _tool_response("relation_add", {
            "subject_uri": _ORPHAN["uri"], "property_uri": _SUBCLASS, "object_value": _NS + "Part",
        }),
        _text_response("Connected Wheel under Part."),
    ])
    session = _FakeSession([(_CLEAN, [_ORPHAN]), (_CLEAN, [])])

    summary, out = _run(session, **_REPAIR_ONLY)

    assert _labels(summary) == ["Round 1", "Round 2"]
    assert summary["checks"][0]["orphans"] == 1 and summary["checks"][1]["orphans"] == 0
    assert summary["changes"] == 1
    assert out["backups"] == [_TTL]
    assert out["exports"] == [_TTL]
    # The backup is taken before the LLM changes anything.
    assert session.names().index("ontology_export") < session.names().index("relation_add")
    # The task message carries the problem and goes to Scaleway with only the fix tools.
    assert "Orphan class" in sent[0]["messages"][1]["content"]
    assert sent[0]["model"] == "scaleway/m"
    assert {t["function"]["name"] for t in sent[0]["tools"]} == FIX_TOOLS


def test_stops_when_a_round_changes_nothing(monkeypatch):
    _script_completions(monkeypatch, [_text_response("Left as is: no support in the text.")])
    summary, out = _run(_FakeSession([(_CLEAN, [_ORPHAN])]), **_REPAIR_ONLY)

    assert _labels(summary) == ["Round 1", "Round 2"]
    assert any("Nothing changed since the last fixes" in line for line in out["logs"])
    # Nothing was changed, so the persisted ontology isn't rewritten.
    assert out["exports"] == []


def test_fix_orphans_false_ignores_orphans(monkeypatch):
    sent = _script_completions(monkeypatch, [])
    session = _FakeSession([(_CLEAN, [_ORPHAN])])

    summary, _ = _run(session, fix_orphans=False, **_REPAIR_ONLY)

    assert sent == []
    assert "ontology_orphans" not in session.names()
    assert summary["checks"][0]["orphans"] == 0


def test_tool_outside_fix_tools_is_refused(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("chunk_mark_processed", {"chunk_id": "c1"}),
        _text_response("Done."),
    ])
    session = _FakeSession([(_CLEAN, [_ORPHAN]), (_CLEAN, [])])

    _run(session, max_rounds=1, **_REPAIR_ONLY)

    assert "chunk_mark_processed" not in session.names()


def test_max_rounds_reached_runs_a_final_check(monkeypatch):
    _script_completions(monkeypatch, [
        _tool_response("relation_delete", {"subject_uri": "a", "property_uri": "p", "object_value": "b"}),
        _text_response("Removed."),
    ])
    session = _FakeSession([(_CLEAN, [_ORPHAN]), (_CLEAN, [])])

    summary, _ = _run(session, max_rounds=1, **_REPAIR_ONLY)

    assert _labels(summary) == ["Round 1", "Round — final"]


def test_unknown_term_is_handed_over_with_its_suggestion(monkeypatch):
    sent = _script_completions(monkeypatch, [_text_response("Fixed.")])
    report = {**_CLEAN, "integrity": {"unknown_terms": [{
        "subject": _NS + "A", "property": "http://www.w3.org/2000/01/rdf-schema#subClassof",
        "object": _NS + "B", "term": "http://www.w3.org/2000/01/rdf-schema#subClassof", "suggestion": _SUBCLASS,
    }]}}

    summary, _ = _run(_FakeSession([(report, []), (_CLEAN, [])]), **_REPAIR_ONLY)

    task = sent[0]["messages"][1]["content"]
    assert "Unknown vocabulary term" in task and f"`{_SUBCLASS}`" in task
    assert summary["checks"][0]["unknown_terms"] == 1


# ── Deduplication ────────────────────────────────────────────────────

def _dedup_handlers(search):
    classes = [{"uri": _NS + "Car", "label": "Car"}, {"uri": _NS + "Automobile", "label": "Automobile"}]
    return {"concept_list": lambda a: json.dumps(classes), "concept_semantic_search": search}


def test_dedup_hands_similar_pairs_to_the_llm(monkeypatch):
    def search(args):
        other = "Automobile" if args["query"].startswith("Car") else "Car"
        return json.dumps([{"uri": _NS + other, "score": 0.95}])

    sent = _script_completions(monkeypatch, [
        _tool_response("concept_merge", {"keep_uri": _NS + "Car", "merge_uri": _NS + "Automobile"}),
        _text_response("Merged."),
    ])
    session = _FakeSession([(_CLEAN, [])], _dedup_handlers(search))

    summary, out = _run(session, dedup=True, enrich_disjointness=False, infer=False)

    assert summary["dedup_pairs"] == 1  # the pair is found from both sides, reviewed once
    assert "duplicate candidates" in sent[0]["messages"][1]["content"]
    assert "concept_merge" in session.names()
    assert out["backups"] == [_TTL]


def test_dedup_skipped_without_embeddings(monkeypatch):
    sent = _script_completions(monkeypatch, [])
    session = _FakeSession([(_CLEAN, [])], _dedup_handlers(lambda a: "Error: embeddings disabled"))

    summary, out = _run(session, dedup=True, enrich_disjointness=False, infer=False)

    assert sent == []
    assert summary["dedup_pairs"] == 0
    assert any("Deduplication skipped" in line for line in out["logs"])


# ── Enrichment ───────────────────────────────────────────────────────

class _Disjointness:
    """sparql_query for two root classes, and disjoint_add recording the declared pairs."""

    def __init__(self):
        self.pairs: list[tuple[str, str]] = []

    def sparql(self, args):
        q = args["query"]
        if "?c ?parent" in q:
            return json.dumps({"rows": [{"c": _NS + "Person"}, {"c": _NS + "Document"}]})
        if "SELECT ?a ?b" in q and "UNION" not in q:  # the declared disjoint pairs
            return json.dumps({"rows": [{"a": a, "b": b} for a, b in self.pairs]})
        return json.dumps({"rows": []})

    def disjoint_add(self, args):
        self.pairs.append(tuple(args["class_uris"]))
        return "ok"


def test_enrichment_declares_disjoint_siblings_then_checks_again(monkeypatch):
    d = _Disjointness()
    sent = _script_completions(monkeypatch, [
        _tool_response("disjoint_add", {"class_uris": [_NS + "Document", _NS + "Person"]}),
        _text_response("Person / Document: a person is never a document."),
    ])
    session = _FakeSession([(_CLEAN, [])], {"sparql_query": d.sparql, "disjoint_add": d.disjoint_add})

    summary, out = _run(session, dedup=False, enrich_disjointness=True, infer=False)

    assert "declare disjoint sibling classes" in sent[0]["messages"][1]["content"]
    assert summary["disjoint_added"] == 1
    assert _labels(summary) == ["Round 1", "After enrichment 1"]
    assert out["exports"] == [_TTL]


def test_disjointness_added_by_the_run_is_flagged_in_explanations(monkeypatch):
    d = _Disjointness()
    unsat = {"reasoner": {
        "consistent": True,
        "unsatisfiable_classes": [{
            "class": "Author", "uri": _NS + "Author",
            "explanations": [["Document disjointWith Person", "Author subClassOf Person", "Author subClassOf Document"]],
        }],
        "entities": {"Author": _NS + "Author", "Person": _NS + "Person", "Document": _NS + "Document"},
    }, "integrity": {}}
    sent = _script_completions(monkeypatch, [
        _tool_response("disjoint_add", {"class_uris": [_NS + "Document", _NS + "Person"]}),
        _text_response("Declared."),
        _text_response("Undone."),
    ])
    session = _FakeSession([(_CLEAN, []), (unsat, []), (_CLEAN, [])],
                           {"sparql_query": d.sparql, "disjoint_add": d.disjoint_add})

    _run(session, dedup=False, enrich_disjointness=True, infer=False)

    fix_task = sent[2]["messages"][1]["content"]
    assert "⚠ declared by the enrichment of this run" in fix_task
    assert f"object_value=`{_NS}Person`" in fix_task


def test_enrichment_skipped_while_inconsistent(monkeypatch):
    inconsistent = {"reasoner": {"consistent": False, "inconsistency": {"reason": "x", "explanations": []},
                                 "unsatisfiable_classes": [], "entities": {}}, "integrity": {}}
    _script_completions(monkeypatch, [_text_response("Cannot decide.")])
    session = _FakeSession([(inconsistent, [])])

    summary, out = _run(session, dedup=False, enrich_disjointness=True, infer=True)

    assert any("enrichment skipped" in line for line in out["logs"])
    assert any("nothing can be inferred" in line for line in out["logs"])
    assert "ontology_infer" not in session.names()
    assert summary["inferences"] is None


# ── Inference ────────────────────────────────────────────────────────

def _infer_handler(items):
    def handler(args):
        if args.get("action") == "materialize":
            return json.dumps({"stored": len(items), "replaced": 0})
        return json.dumps({"count": len(items), "materialized": 0, "inferences": items})
    return handler


_INFERENCE = {
    "subject": _NS + "LlmAgent", "predicate": _SUBCLASS, "object": _NS + "Software", "taxonomic": True,
    "explanations": [["LlmAgent subClassOf LargeLanguageModel", "LargeLanguageModel subClassOf Software"]],
    "entities": {"LlmAgent": _NS + "LlmAgent", "LargeLanguageModel": _NS + "LargeLanguageModel"},
    "materialized": False,
}


def test_inferences_are_reviewed_checked_again_then_materialized(monkeypatch):
    sent = _script_completions(monkeypatch, [_text_response("All axioms hold; deleted nothing.")])
    session = _FakeSession([(_CLEAN, [])], {"ontology_infer": _infer_handler([_INFERENCE])})

    summary, out = _run(session, dedup=False, enrich_disjointness=False, infer=True)

    review_task = sent[0]["messages"][1]["content"]
    assert "review 1 inference(s)" in review_task
    assert "LlmAgent subClassOf LargeLanguageModel" in review_task
    actions = [a.get("action") for n, a in session.calls if n == "ontology_infer"]
    assert actions == ["preview", "materialize"]
    assert _labels(summary) == ["Round 1", "After review 1"]
    assert summary["inferences"] == {"entailed": 1, "taxonomic": 1, "materialized": 1, "replaced": 0}
    assert out["backups"] == [_TTL]
    assert out["exports"] == [_TTL]  # materializing changed the ontology


def test_inferences_materialized_without_review(monkeypatch):
    sent = _script_completions(monkeypatch, [])
    session = _FakeSession([(_CLEAN, [])], {"ontology_infer": _infer_handler([_INFERENCE])})

    summary, _ = _run(session, dedup=False, enrich_disjointness=False, infer=True, review_inferences=False)

    assert sent == []
    assert summary["inferences"]["materialized"] == 1


def test_inference_skipped_when_the_reasoner_is_unavailable(monkeypatch):
    _script_completions(monkeypatch, [])
    session = _FakeSession([(_CLEAN, [])], {"ontology_infer": lambda a: "Error: Java not found"})

    summary, out = _run(session, dedup=False, enrich_disjointness=False, infer=True)

    assert summary["inferences"] is None
    assert any("Inference skipped" in line for line in out["logs"])


# ── Helpers ──────────────────────────────────────────────────────────

def test_report_counts_reports_reasoner_error():
    counts = report_counts(
        {"reasoner": {"error": "Java not found"},
         "integrity": {"domain_violations": [{}, {}], "unknown_terms": [{}]}}, [],
    )
    assert counts["consistent"] is None
    assert counts["reasoner_error"] == "Java not found"
    assert counts["domain"] == 2
    assert counts["unknown_terms"] == 1


def test_prune_history_keeps_recent_turns_and_task_message():
    long = "x" * 1000
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": long}]
    for i in range(3):
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": str(i)}]})
        messages.append({"role": "tool", "tool_call_id": str(i), "content": long})

    pruned = prune_history(messages, keep_recent_turns=1, pruned_chars=10)

    assert pruned[1]["content"] == long                 # task message untouched
    assert pruned[3]["content"].startswith("x" * 10 + "\n… [pruned")
    assert pruned[-1]["content"] == long                # last turn kept whole
    assert messages[3]["content"] == long               # original not mutated
