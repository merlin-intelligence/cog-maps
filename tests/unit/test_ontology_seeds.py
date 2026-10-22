"""Unit tests for cogmaps.ontology.seeds: seed loading through OLAF's CLI, and the
migration of an ontology's semantic index.

OLAF, Qdrant and Oxigraph are faked — these tests check what is asked of them.
"""
from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace

import pytest

from cogmaps.ontology import olaf_cli, seeds


@pytest.fixture(autouse=True)
def _no_scratch_graph_cleanup(monkeypatch):
    monkeypatch.setattr(olaf_cli, "_drop_scratch_graph", lambda: None)


@pytest.fixture
def olaf_calls(monkeypatch):
    """Record the ``olaf`` commands run, with the content of the file they load, if any."""
    calls: list[dict] = []

    def fake_run_olaf(args, config_dir, *, timeout=None):
        assert os.path.exists(os.path.join(config_dir, "config.toml"))
        source = args[1] if len(args) > 1 else ""
        data = pathlib.Path(source).read_bytes() if os.path.isfile(source) else None
        calls.append({"args": args, "data": data})
        return SimpleNamespace(stdout="Seed 'x' chargée.\n", stderr="")

    monkeypatch.setattr(seeds, "run_olaf", fake_run_olaf)
    return calls


class _FakeQdrant:
    def __init__(self, counts: dict[str, int]):
        self.counts = dict(counts)
        self.deleted: list[str] = []

    def collection_exists(self, name):
        return name in self.counts

    def count(self, name):
        return SimpleNamespace(count=self.counts[name])

    def delete_collection(self, name):
        self.deleted.append(name)
        del self.counts[name]


# ── Loading ──────────────────────────────────────────────────────────

def test_load_seed_from_file_keeps_the_file_name(olaf_calls):
    report = seeds.load_seed_from_file(b"@prefix : <x#> .", "fibo-be.ttl", qdrant_url="http://q:6333")

    (call,) = olaf_calls
    assert call["args"][0] == "load-seed"
    assert os.path.basename(call["args"][1]) == "fibo-be.ttl"  # OLAF derives the id and format from it
    assert call["args"][2:] == []
    assert call["data"] == b"@prefix : <x#> ."
    assert "chargée" in report


def test_load_seed_from_file_ignores_directories_in_the_upload_name(olaf_calls):
    seeds.load_seed_from_file(b"", "../../etc/onto.owl", qdrant_url="http://q:6333")
    assert os.path.basename(olaf_calls[0]["args"][1]) == "onto.owl"
    assert "etc" not in olaf_calls[0]["args"][1]


def test_load_seed_from_url_passes_id_and_format(olaf_calls):
    seeds.load_seed_from_url("https://example.org/onto", qdrant_url="http://q:6333", seed_id="ex", fmt="xml")
    assert olaf_calls[0]["args"] == ["load-seed", "https://example.org/onto", "--id", "ex", "--format", "xml"]


def test_index_seed_runs_index_seed(olaf_calls):
    seeds.index_seed("fibo", qdrant_url="http://q:6333")
    assert olaf_calls[0]["args"] == ["index-seed", "fibo"]


def test_list_seeds_reads_graphs_and_index_sizes(monkeypatch):
    monkeypatch.setattr(seeds, "is_reachable", lambda url: True)
    monkeypatch.setattr(seeds, "run_query", lambda url, q: ("table", {"results": {"bindings": [
        {"g": {"value": "urn:olaf:seed:fibo"}, "triples": {"value": "1200"}},
        {"g": {"value": "urn:olaf:seed:old"}, "triples": {"value": "10"}},
    ]}}))
    qdrant = _FakeQdrant({"olaf_index_seed_fibo": 340})

    assert seeds.list_seeds(qdrant) == [
        seeds.SeedInfo("fibo", 1200, 340),
        seeds.SeedInfo("old", 10, None),
    ]


# ── Semantic index migration ─────────────────────────────────────────

def _ensure(monkeypatch, olaf_calls, qdrant, *, has_entities=True):
    monkeypatch.setattr(seeds, "_has_entities", lambda oid: has_entities)
    logs: list[str] = []
    with olaf_cli.scratch_config_dir(qdrant_url="http://q:6333", ontology_id="main") as config_dir:
        seeds.ensure_ontology_index("main", config_dir, qdrant, logs.append)
    return [c["args"] for c in olaf_calls], logs


def test_legacy_concepts_collection_is_reindexed_then_dropped(monkeypatch, olaf_calls):
    qdrant = _FakeQdrant({"olaf_concepts_main": 12})
    calls, _ = _ensure(monkeypatch, olaf_calls, qdrant)
    assert calls == [["reindex", "main"]]
    assert qdrant.deleted == ["olaf_concepts_main"]


def test_empty_index_of_a_populated_ontology_is_rebuilt(monkeypatch, olaf_calls):
    # A search session creates the index collection empty when it is missing.
    calls, _ = _ensure(monkeypatch, olaf_calls, _FakeQdrant({"olaf_index_main": 0}))
    assert calls == [["reindex", "main"]]


def test_existing_index_is_left_alone(monkeypatch, olaf_calls):
    qdrant = _FakeQdrant({"olaf_index_main": 30})
    calls, _ = _ensure(monkeypatch, olaf_calls, qdrant)
    assert calls == []
    assert qdrant.deleted == []


def test_ontology_without_entities_is_not_reindexed(monkeypatch, olaf_calls):
    calls, _ = _ensure(monkeypatch, olaf_calls, _FakeQdrant({}), has_entities=False)
    assert calls == []
