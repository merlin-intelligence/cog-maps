"""Unit tests for cogmaps.ontology.oxigraph_client: the read-only SPARQL guard
and query-kind dispatch (table vs turtle), with Oxigraph's HTTP calls mocked out.
"""
from __future__ import annotations

import pytest

from cogmaps.ontology import oxigraph_client as ox


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, json_data=None, text_data=""):
        self.status_code = status_code
        self.headers = headers or {}
        self._json_data = json_data
        self.text = text_data

    def json(self):
        return self._json_data


@pytest.mark.parametrize("keyword", ["INSERT", "DELETE", "DROP", "CLEAR", "LOAD", "CREATE"])
def test_run_query_rejects_update_keywords(keyword, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("requests.post should not be called for a rejected query")

    monkeypatch.setattr(ox.requests, "post", _fail_if_called)
    with pytest.raises(ox.SparqlError, match="read-only"):
        ox.run_query("http://oxigraph.example", f"{keyword} DATA {{ <a> <b> <c> }}")


def test_run_query_allows_select_without_network_error_about_update(monkeypatch):
    def _fake_post(url, data=None, headers=None, timeout=None):
        return _FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/sparql-results+json"},
            json_data={"head": {"vars": ["s"]}, "results": {"bindings": []}},
        )

    monkeypatch.setattr(ox.requests, "post", _fake_post)
    kind, payload = ox.run_query("http://oxigraph.example", "SELECT ?s WHERE { ?s ?p ?o }")
    assert kind == "table"
    assert payload["results"]["bindings"] == []


def test_run_query_returns_turtle_for_construct(monkeypatch):
    def _fake_post(url, data=None, headers=None, timeout=None):
        return _FakeResponse(
            status_code=200,
            headers={"Content-Type": "text/turtle"},
            text_data="<a> <b> <c> .",
        )

    monkeypatch.setattr(ox.requests, "post", _fake_post)
    kind, payload = ox.run_query("http://oxigraph.example", "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
    assert kind == "turtle"
    assert payload == "<a> <b> <c> ."


def test_run_query_raises_on_non_200(monkeypatch):
    def _fake_post(url, data=None, headers=None, timeout=None):
        return _FakeResponse(status_code=400, text_data="malformed query")

    monkeypatch.setattr(ox.requests, "post", _fake_post)
    with pytest.raises(ox.SparqlError, match="malformed query"):
        ox.run_query("http://oxigraph.example", "SELECT ?s WHERE { ?s ?p ?o }")


def test_list_graphs_extracts_uris_from_bindings(monkeypatch):
    def _fake_post(url, data=None, headers=None, timeout=None):
        return _FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/sparql-results+json"},
            json_data={
                "head": {"vars": ["g"]},
                "results": {"bindings": [
                    {"g": {"value": "urn:olaf:main"}},
                    {"g": {"value": "urn:olaf:seed:core"}},
                ]},
            },
        )

    monkeypatch.setattr(ox.requests, "post", _fake_post)
    assert ox.list_graphs("http://oxigraph.example") == ["urn:olaf:main", "urn:olaf:seed:core"]


def test_export_graph_ttl_returns_turtle_text(monkeypatch):
    def _fake_post(url, data=None, headers=None, timeout=None):
        assert "urn:olaf:main" in data.decode("utf-8")
        return _FakeResponse(status_code=200, headers={"Content-Type": "text/turtle"}, text_data="<a> <b> <c> .")

    monkeypatch.setattr(ox.requests, "post", _fake_post)
    assert ox.export_graph_ttl("http://oxigraph.example", "urn:olaf:main") == "<a> <b> <c> ."


def test_is_reachable_false_on_connection_error(monkeypatch):
    def _fake_get(url, timeout=None):
        raise ox.requests.RequestException("connection refused")

    monkeypatch.setattr(ox.requests, "get", _fake_get)
    assert ox.is_reachable("http://oxigraph.example") is False
