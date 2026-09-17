"""Read-only SPARQL 1.1 client for Oxigraph, used by the Ontology Explorer page.

Talks to Oxigraph's ``/query`` HTTP endpoint directly — no OLAF/MCP subprocess
involved. Browsing and querying an already-built ontology needs neither the
chunk-processing agent loop nor a spawned ``olaf`` process, both of which are
scoped to building (:mod:`cogmaps.ontology.runner`).

Deliberately never touches Oxigraph's ``/update`` endpoint — this client can
only run SPARQL *queries* (SELECT/ASK/CONSTRUCT/DESCRIBE).
"""
from __future__ import annotations

import re

import requests

_TIMEOUT = 30
_UPDATE_KEYWORD_RE = re.compile(
    r"\b(INSERT|DELETE|DROP|CLEAR|LOAD|CREATE|COPY|MOVE|ADD)\b", re.IGNORECASE
)


class SparqlError(RuntimeError):
    """Raised when Oxigraph is unreachable or rejects a query."""


def is_reachable(oxigraph_url: str) -> bool:
    try:
        r = requests.get(oxigraph_url.rstrip("/") + "/", timeout=5)
        return r.status_code < 500
    except requests.RequestException:
        return False


def _guard_read_only(query: str) -> None:
    """Reject anything that looks like a SPARQL Update, as a UX safeguard.

    Oxigraph's ``/query`` endpoint only executes SPARQL Query forms anyway
    (Updates require ``/update``) — this just turns a would-be protocol
    error into a clear, specific message before the request is even sent.
    A false positive on a keyword inside a string literal is possible; that
    trade-off is fine here since it's a UX nicety, not the actual boundary.
    """
    m = _UPDATE_KEYWORD_RE.search(query)
    if m:
        raise SparqlError(
            f"This page only runs read-only SPARQL queries (SELECT/ASK/CONSTRUCT/DESCRIBE) "
            f"— {m.group(1)!r} looks like an update operation and was rejected."
        )


def run_query(oxigraph_url: str, query: str) -> tuple[str, dict | str]:
    """Run a read-only SPARQL query against Oxigraph.

    Returns ``("table", sparql_json_results)`` for SELECT/ASK, or
    ``("turtle", turtle_text)`` for CONSTRUCT/DESCRIBE — determined from the
    response's actual ``Content-Type`` rather than parsing the query text,
    since Oxigraph itself decides the result form.
    """
    _guard_read_only(query)
    try:
        r = requests.post(
            oxigraph_url.rstrip("/") + "/query",
            data=query.encode("utf-8"),
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json, text/turtle;q=0.9",
            },
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        raise SparqlError(f"Cannot reach Oxigraph: {e}") from e
    if r.status_code != 200:
        raise SparqlError(r.text.strip() or f"Oxigraph returned HTTP {r.status_code}")

    content_type = r.headers.get("Content-Type", "")
    if "json" in content_type:
        return "table", r.json()
    return "turtle", r.text


def list_graphs(oxigraph_url: str) -> list[str]:
    """Distinct named graph URIs currently in the store."""
    kind, payload = run_query(
        oxigraph_url, "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } } ORDER BY ?g"
    )
    if kind != "table":
        raise SparqlError("Unexpected response listing graphs.")
    return [row["g"]["value"] for row in payload.get("results", {}).get("bindings", [])]


def export_graph_ttl(oxigraph_url: str, graph_uri: str) -> str:
    """CONSTRUCT-export one named graph's triples as Turtle."""
    query = f"CONSTRUCT {{ ?s ?p ?o }} WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}"
    kind, payload = run_query(oxigraph_url, query)
    if kind != "turtle":
        raise SparqlError("Unexpected response exporting the graph.")
    return payload
