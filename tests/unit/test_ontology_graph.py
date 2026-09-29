"""Unit tests for cogmaps.ontology.graph.build_pyvis_html: Turtle -> navigable pyvis graph."""
from __future__ import annotations

from cogmaps.ontology.graph import build_pyvis_html

_TTL = """
@prefix : <http://olaf.local/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Vehicle a owl:Class ; rdfs:label "Vehicle" .
:Car a owl:Class ; rdfs:label "Car" ; rdfs:subClassOf :Vehicle .
:Engine a owl:Class ; rdfs:label "Engine" .

:hasEngine a owl:ObjectProperty ; rdfs:label "hasEngine" ;
    rdfs:domain :Car ; rdfs:range :Engine .

:Car :hasEngine :Engine .

:TeslaModel3 a owl:NamedIndividual ; rdfs:label "Tesla Model 3" ; a :Car .

:Car <urn:olaf:extractedFrom> <urn:olaf:chunk:chunk-42> .
"""


def test_build_pyvis_html_writes_a_file(tmp_path):
    out = tmp_path / "graph.html"
    result = build_pyvis_html(_TTL, str(out))
    assert result == str(out)
    assert out.exists()


def test_build_pyvis_html_includes_class_and_property_labels(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_TTL, str(out))
    content = out.read_text(encoding="utf-8")
    assert "Vehicle" in content
    assert "Car" in content
    assert "Engine" in content
    assert "hasEngine" in content
    assert "subClassOf" in content


def test_build_pyvis_html_surfaces_source_chunk_in_tooltip(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_TTL, str(out))
    content = out.read_text(encoding="utf-8")
    assert "chunk-42" in content


def test_build_pyvis_html_shows_physics_controls_by_default(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_TTL, str(out))
    content = out.read_text(encoding="utf-8")
    assert '"configure": {"enabled": true' in content


def test_build_pyvis_html_can_hide_physics_controls(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_TTL, str(out), show_physics_controls=False)
    content = out.read_text(encoding="utf-8")
    assert '"configure": {"enabled": false' in content


def test_build_pyvis_html_matches_body_background_to_avoid_blank_chrome(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_TTL, str(out))
    content = out.read_text(encoding="utf-8")
    assert "html,body{margin:0;padding:0;background:#faf6f0;}" in content
    assert ".card{background:#faf6f0 !important;border:none !important;}" in content


def test_build_pyvis_html_handles_empty_ontology(tmp_path):
    out = tmp_path / "empty.html"
    build_pyvis_html("", str(out))
    assert out.exists()


def test_build_pyvis_html_renders_schema_domain_range_edges_and_external_parents(tmp_path):
    ttl = """
    @prefix : <http://olaf.local/ontology#> .
    @prefix seed: <http://example.org/ontology#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

    :TreasuryBond a owl:Class ;
        rdfs:label "Treasury Bond" ;
        skos:definition "A sovereign bond issued by the Treasury." ;
        rdfs:altLabel "T-Bond" ;
        rdfs:subClassOf seed:Asset .

    :Yield a owl:Class ;
        rdfs:label "Yield" .

    :hasYield a owl:ObjectProperty ;
        rdfs:label "hasYield" ;
        rdfs:domain :TreasuryBond ;
        rdfs:range :Yield .
    """
    out = tmp_path / "schema_graph.html"
    build_pyvis_html(ttl, str(out))
    content = out.read_text(encoding="utf-8")

    # Parent seed class should be included as a node
    assert "Asset" in content
    assert "Treasury Bond" in content
    assert "Yield" in content

    # Schema-level domain/range edge and subClassOf edge should exist
    assert "hasYield" in content
    assert "subClassOf" in content

    # SKOS definition and alt label should appear in tooltip
    assert "A sovereign bond issued by the Treasury." in content
    assert "alt: T-Bond" in content
