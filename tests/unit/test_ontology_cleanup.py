"""Unit tests for cogmaps.ontology.cleanup: ontology existence check + cascade delete.

All external calls (subprocess, Qdrant, Oxigraph) are mocked — these tests
verify the orchestration logic, not a live OLAF/Qdrant/Oxigraph stack.
"""
from __future__ import annotations

import pytest

from cogmaps.ontology import cleanup, olaf_cli


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeQdrantStore:
    def __init__(self, collections):
        self._collections = list(collections)
        self.deleted = []

    def list_collections(self):
        return self._collections

    def delete_collection(self, name):
        self.deleted.append(name)
        self._collections.remove(name)


@pytest.fixture(autouse=True)
def _no_scratch_graph_cleanup(monkeypatch):
    """The scratch-graph cleanup talks to Oxigraph: not under test here."""
    monkeypatch.setattr(olaf_cli, "_drop_scratch_graph", lambda: None)


def _olaf_args(cmd: list[str]) -> list[str]:
    """The ``olaf`` arguments of a command run through the launcher."""
    assert cmd[1:3] == ["-m", "cogmaps.ontology.olaf_launcher"]
    return cmd[3:]


@pytest.fixture(autouse=True)
def _isolated_profiles_dir(tmp_path, monkeypatch):
    """Keep blueprint deletion away from the real user_data/ directory."""
    profiles = tmp_path / "domain_profiles"
    monkeypatch.setenv("DOMAIN_PROFILES_DIR", str(profiles))
    return profiles


def test_ontology_ids_for_empty_when_oxigraph_unreachable(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: False)
    assert cleanup.ontology_ids_for("my_collection") == []


def test_ontology_ids_for_lists_the_collection_ontologies_default_first(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: True)
    monkeypatch.setattr(cleanup, "list_graphs", lambda url: [
        "urn:olaf:my_collection__v2", "urn:olaf:my_collection", "urn:olaf:seed:core",
        "urn:olaf:my_collection_other", "urn:olaf:other",
    ])
    assert cleanup.ontology_ids_for("my_collection") == ["my_collection", "my_collection__v2"]


def test_ontology_ids_for_empty_when_graph_absent(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: True)
    monkeypatch.setattr(cleanup, "list_graphs", lambda url: ["urn:olaf:other"])
    assert cleanup.ontology_ids_for("my_collection") == []


def test_ontology_ids_for_empty_on_query_error(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: True)

    def _raising_list_graphs(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(cleanup, "list_graphs", _raising_list_graphs)
    assert cleanup.ontology_ids_for("my_collection") == []


def test_delete_ontology_drops_graph_resets_chunks_and_cleans_up(tmp_path, monkeypatch, _isolated_profiles_dir):
    commands = []

    def _fake_run(cmd, **kwargs):
        commands.append(_olaf_args(cmd))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(olaf_cli.subprocess, "run", _fake_run)

    # The olaf_index_ collection is OLAF's to drop; only a legacy olaf_concepts_ one is left to cog-maps.
    fake_store = _FakeQdrantStore(collections=["my_collection", "olaf_concepts_my_collection__v2"])
    monkeypatch.setattr(cleanup, "QdrantStore", lambda host, port: fake_store)

    ttl_file = tmp_path / "v2.ttl"
    ttl_file.write_text("<a> <b> <c> .", encoding="utf-8")
    monkeypatch.setattr(cleanup, "ttl_path_for", lambda collection, ontology_id: str(ttl_file))
    _isolated_profiles_dir.mkdir()
    blueprint_file = _isolated_profiles_dir / "my_collection.json"
    blueprint_file.write_text("{}", encoding="utf-8")

    messages = cleanup.delete_ontology("my_collection", "my_collection__v2", "localhost", 6333)

    assert commands == [["drop", "my_collection__v2", "--collection", "my_collection"]]
    assert fake_store.deleted == ["olaf_concepts_my_collection__v2"]
    assert not ttl_file.exists()
    assert blueprint_file.exists()  # shared by the collection's other ontologies
    assert any("dropped" in m for m in messages)
    assert any("olaf_concepts_my_collection__v2" in m for m in messages)
    assert any("Turtle" in m for m in messages)


def test_delete_ontologies_for_collection_drops_each_ontology_and_the_blueprint(
    tmp_path, monkeypatch, _isolated_profiles_dir,
):
    commands = []

    def _fake_run(cmd, **kwargs):
        commands.append(_olaf_args(cmd))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(olaf_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(cleanup, "QdrantStore", lambda host, port: _FakeQdrantStore(collections=["my_collection"]))
    monkeypatch.setattr(cleanup, "ttl_path_for", lambda collection, ontology_id: str(tmp_path / "absent.ttl"))
    monkeypatch.setattr(cleanup, "ontology_ids_for", lambda collection: ["my_collection", "my_collection__v2"])

    _isolated_profiles_dir.mkdir()
    blueprint_file = _isolated_profiles_dir / "my_collection.json"
    blueprint_file.write_text("{}", encoding="utf-8")
    other_file = _isolated_profiles_dir / "other_collection.json"
    other_file.write_text("{}", encoding="utf-8")

    messages = cleanup.delete_ontologies_for_collection("my_collection", "localhost", 6333)

    assert [c[:2] for c in commands] == [["drop", "my_collection"], ["drop", "my_collection__v2"]]
    assert not blueprint_file.exists()
    assert other_file.exists()
    assert any("blueprint" in m for m in messages)


def test_delete_ontology_skips_absent_legacy_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(olaf_cli.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(returncode=0))

    fake_store = _FakeQdrantStore(collections=["my_collection"])
    monkeypatch.setattr(cleanup, "QdrantStore", lambda host, port: fake_store)
    monkeypatch.setattr(cleanup, "ttl_path_for", lambda collection, ontology_id: str(tmp_path / "absent.ttl"))

    messages = cleanup.delete_ontology("my_collection", "my_collection", "localhost", 6333)

    assert fake_store.deleted == []
    assert not any("olaf_concepts" in m for m in messages)


def test_delete_ontology_raises_on_subprocess_failure(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stderr="graph not found")

    monkeypatch.setattr(olaf_cli.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="graph not found"):
        cleanup.delete_ontology("my_collection", "my_collection", "localhost", 6333)


def test_list_seed_ids_empty_when_oxigraph_unreachable(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: False)
    assert cleanup.list_seed_ids() == []


def test_list_seed_ids_filters_and_strips_prefix(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: True)
    monkeypatch.setattr(
        cleanup, "list_graphs",
        lambda url: ["urn:olaf:main", "urn:olaf:seed:core", "urn:olaf:seed:legal"],
    )
    assert cleanup.list_seed_ids() == ["core", "legal"]


def test_list_seed_ids_empty_on_query_error(monkeypatch):
    monkeypatch.setattr(cleanup, "is_reachable", lambda url: True)

    def _raising_list_graphs(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(cleanup, "list_graphs", _raising_list_graphs)
    assert cleanup.list_seed_ids() == []


def test_delete_seed_runs_olaf_drop_seed(monkeypatch):
    captured_cmd = {}

    def _fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(olaf_cli.subprocess, "run", _fake_run)

    messages = cleanup.delete_seed("core", "localhost", 6333)

    assert _olaf_args(captured_cmd["cmd"]) == ["drop-seed", "core"]
    assert any("core" in m and "dropped" in m for m in messages)


def test_delete_seed_raises_on_subprocess_failure(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stderr="seed not found")

    monkeypatch.setattr(olaf_cli.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="seed not found"):
        cleanup.delete_seed("nonexistent", "localhost", 6333)
