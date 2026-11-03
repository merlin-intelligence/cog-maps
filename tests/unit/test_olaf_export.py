"""Unit tests for the ontology export helpers in cogmaps.ontology.olaf_config:
the rdflib-based content check and the direct Oxigraph export."""
from __future__ import annotations

import io
import logging
import urllib.error

import cogmaps.ontology.olaf_config as olaf_config
from cogmaps.ontology.olaf_config import export_oxigraph_ontology_ttl, has_ontology_content

_PREFIXED = "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n<http://olaf.local/ontology#Car> a owl:Class ."
_FULL_IRIS = "<http://olaf.local/ontology#Car> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#Class> ."


def test_has_ontology_content_accepts_classes_in_any_turtle_form():
    assert has_ontology_content(_PREFIXED)
    assert has_ontology_content(_FULL_IRIS)


def test_has_ontology_content_accepts_properties_and_individuals():
    ttl = (
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://olaf.local/ontology#manages> a owl:ObjectProperty .\n"
        "<http://olaf.local/ontology#ACME> a owl:NamedIndividual ."
    )
    assert has_ontology_content(ttl)


def test_has_ontology_content_rejects_empty_invalid_or_error_text():
    assert not has_ontology_content(None)
    assert not has_ontology_content("")
    assert not has_ontology_content("# nothing here\n")
    assert not has_ontology_content("Error: tool failed")
    assert not has_ontology_content('<http://x#a> <http://x#b> "c" .')  # triples, but no ontology terms


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_export_oxigraph_returns_body(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["timeout"] = req.full_url, timeout
        return _Resp(_PREFIXED.encode())

    monkeypatch.setattr(olaf_config.urllib.request, "urlopen", fake_urlopen)
    assert export_oxigraph_ontology_ttl("my_onto", "http://ox:7878/", timeout=3) == _PREFIXED
    assert seen == {"url": "http://ox:7878/store?graph=urn:olaf:my_onto", "timeout": 3}


def test_export_oxigraph_missing_graph_is_quiet(monkeypatch, caplog):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(olaf_config.urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level(logging.WARNING, logger="cogmaps.ontology.olaf_config"):
        assert export_oxigraph_ontology_ttl("my_onto", "http://ox:7878") is None
    assert caplog.records == []


def test_export_oxigraph_unreachable_is_logged(monkeypatch, caplog):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(olaf_config.urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level(logging.WARNING, logger="cogmaps.ontology.olaf_config"):
        assert export_oxigraph_ontology_ttl("my_onto", "http://ox:7878") is None
    assert any("connection refused" in r.getMessage() for r in caplog.records)


def test_ontology_ids_of_a_collection():
    assert olaf_config.ontology_id_for("My-Col") == "my_col"
    assert olaf_config.ontology_id_for("My-Col", "  ") == "my_col"
    assert olaf_config.ontology_id_for("My-Col", "Legal v2") == "my_col__legal_v2"
    assert olaf_config.ontology_name_of("My-Col", "my_col__legal_v2") == "legal_v2"
    assert olaf_config.ontology_name_of("My-Col", "my_col") == ""
    assert olaf_config.ontology_display_name("My-Col", "my_col__legal_v2") == "My-Col — legal_v2"
    assert olaf_config.ontology_display_name("My-Col", "my_col") == "My-Col"
    graphs = ["urn:olaf:my_col__b", "urn:olaf:my_col__a", "urn:olaf:my_col", "urn:olaf:my_col_x", "urn:olaf:seed:s"]
    assert olaf_config.collection_ontology_ids("My-Col", graphs) == ["my_col", "my_col__a", "my_col__b"]
    assert olaf_config.collection_ontology_ids("My-Col", ["urn:olaf:my_col__a"]) == ["my_col__a"]


def test_default_ontology_keeps_its_local_files_named_after_the_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(olaf_config, "USER_DATA_DIR", tmp_path)
    assert olaf_config.ttl_path_for("My-Col") == str(tmp_path / "ontology" / "My-Col.ttl")
    assert olaf_config.ttl_path_for("My-Col", "my_col") == str(tmp_path / "ontology" / "My-Col.ttl")
    assert olaf_config.ttl_path_for("My-Col", "my_col__v2") == str(tmp_path / "ontology" / "my_col__v2.ttl")
    olaf_config.save_backup("My-Col", "my_col__v2", "<a> <b> <c> .")
    assert olaf_config.list_backups("My-Col") == []
    assert len(olaf_config.list_backups("My-Col", "my_col__v2")) == 1
