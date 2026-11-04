"""Unit tests for the ontology merging agent (cogmaps.ontology.merging_agent).

litellm and the OLAF MCP session are faked out — this tests the agent's own
control flow (import → match → decide → merge → alignment), not the LLM or OLAF.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm
import pytest

from cogmaps.ontology.merging_agent import run_merge

_TTL = "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<http://olaf.local/ontology#Fund> a owl:Class ."
_NS = "http://olaf.local/ontology#"


def _entity(name: str, label: str, source_uri: str | None = None, kind: str = "class") -> dict:
    return {"uri": _NS + name, "source_uri": source_uri or _NS + name.split("_")[0], "kind": kind,
            "label": label, "definition": f"A {label.lower()}.", "source_chunk_id": f"chunk-{name}"}


_REFERENCE = [_entity("Fund", "Fund"), _entity("Programme", "Programme")]
# "Fund" is already used by the reference: renamed on import.
_OTHER = [_entity("Fund_ipcc", "Fund"), _entity("Mangrove", "Mangrove Restoration")]


class _FakeSession:
    def __init__(self, target_has_content: bool = False):
        self.target_has_content = target_has_content
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        if name == "ontology_list":
            text = json.dumps([{"id": "gcf"}, {"id": "ipcc"}, {"id": "target"}])
        elif name == "sparql_query":
            text = json.dumps({"boolean": self.target_has_content} if "ASK" in args["query"] else {"rows": []})
        elif name == "seed_list":
            text = "[]"
        elif name == "ontology_import":
            if args["source_ontology_id"] == "gcf":
                text = json.dumps({"triples": 10, "renamed": {}, "entities": _REFERENCE})
            else:
                text = json.dumps({"triples": 8, "renamed": {_NS + "Fund": _NS + "Fund_ipcc"}, "entities": _OTHER})
        elif name == "concept_semantic_search":
            hits = [{"uri": _NS + "Fund", "score": 0.9}, {"uri": _NS + "Programme", "score": 0.75}]
            if args["query"].startswith("Fund"):
                hits.append({"uri": _NS + "Fund_ipcc", "score": 1.0})  # itself: not a candidate
            text = json.dumps(hits)
        elif name == "chunk_read_batch":
            text = json.dumps([{"id": cid, "text": f"text of {cid}"} for cid in args["chunk_ids"]])
        elif name == "ontology_check":
            text = json.dumps({"reasoner": {"consistent": True}, "integrity": {}, "issues": 0})
        elif name == "ontology_export":
            text = _TTL
        else:
            text = "{}"
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _answer(decisions: list[dict] | str):
    content = decisions if isinstance(decisions, str) else json.dumps({"decisions": decisions})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])


def _script_completions(monkeypatch, responses: list):
    sent: list[dict] = []
    it = iter(responses)

    def fake_completion(**kwargs):
        sent.append(kwargs)
        return next(it)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return sent


def _run(session):
    out = {"alignments": [], "exports": [], "logs": []}
    summary = asyncio.run(run_merge(
        session,
        model="m",
        api_key="k",
        ontology_id="target",
        reference_id="gcf",
        other_id="ipcc",
        reference_collection="gcf_chunks",
        other_collection="ipcc_chunks",
        log=out["logs"].append,
        set_progress=lambda cur, tot: None,
        alignment_cb=out["alignments"].append,
        export_cb=out["exports"].append,
    ))
    return summary, out


def test_merges_confident_equivalence_and_lists_the_rest(monkeypatch):
    sent = _script_completions(monkeypatch, [_answer([
        # O1 = Fund_ipcc (same name first), O2 = Mangrove
        {"other": "O1", "reference": "R1.1", "relation": "equivalent", "confidence": 0.95, "reason": "same fund"},
        {"other": "O2", "reference": "R2.2", "relation": "narrower", "confidence": 0.5, "reason": "a guess"},
    ])])
    session = _FakeSession()

    summary, out = _run(session)

    assert len(sent) == 1
    task = sent[0]["messages"][1]["content"]
    assert "R1.1 (same name, score 0.90)" in task
    assert "Fund_ipcc` " not in task.split("Candidates:")[1].split("### O2")[0]  # never its own candidate
    assert "text of chunk-Mangrove" in task

    assert ("concept_merge", {"keep_uri": _NS + "Fund", "merge_uri": _NS + "Fund_ipcc"}) in session.calls
    assert "relation_add" not in session.names()  # below min confidence
    assert summary["equivalent"] == 1 and summary["subsumptions"] == 0 and summary["not_applied"] == 1
    assert summary["imported"] == {"reference": 2, "other": 2}
    assert summary["check"]["consistent"] is True
    assert [c["applied"] for c in summary["correspondences"]] == [False, True]

    # The alignment is between the URIs of the source ontologies.
    (tsv,) = out["alignments"]
    rows = [r.split("\t") for r in tsv.splitlines() if not r.startswith("#")][1:]
    assert [(r[0], r[2], r[3], r[7]) for r in rows] == [
        (_NS + "Programme", "skos:narrowMatch", _NS + "Mangrove", "false"),
        (_NS + "Fund", "skos:exactMatch", _NS + "Fund", "true"),
    ]
    assert out["exports"] == [_TTL]


def test_refuses_a_target_with_content(monkeypatch):
    _script_completions(monkeypatch, [])
    session = _FakeSession(target_has_content=True)

    with pytest.raises(RuntimeError, match="already has content"):
        _run(session)
    assert "ontology_import" not in session.names()


def test_asks_again_after_an_invalid_answer(monkeypatch):
    sent = _script_completions(monkeypatch, [
        _answer("I think they are the same."),
        _answer([{"other": "O1", "reference": "R1.1", "relation": "equivalent", "confidence": 0.9, "reason": "x"}]),
    ])

    summary, _ = _run(_FakeSession())

    assert len(sent) == 2
    assert sent[1]["messages"][-1]["content"].startswith("Answer with the JSON object only")
    assert summary["equivalent"] == 1
