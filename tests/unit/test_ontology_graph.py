"""Unit tests for cogmaps.ontology.graph.build_pyvis_html: Turtle -> navigable pyvis graph."""
from __future__ import annotations

from cogmaps.ontology.graph import HIGHLIGHT_COLOR, build_pyvis_html, extract_subgraph

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
        skos:altLabel "T-Bond" ;
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


def test_build_pyvis_html_does_not_draw_owl_thing_as_a_node(tmp_path):
    ttl = """
    @prefix : <http://olaf.local/ontology#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Root a owl:Class ; rdfs:label "Root" ; rdfs:subClassOf owl:Thing .
    :rel a owl:ObjectProperty ; rdfs:label "rel" ; rdfs:domain :Root ; rdfs:range rdfs:Resource .
    """
    out = tmp_path / "builtins.html"
    build_pyvis_html(ttl, str(out))
    content = out.read_text(encoding="utf-8")
    assert "Root" in content
    assert "owl#Thing" not in content
    assert "rdf-schema#Resource" not in content


# ── extract_subgraph: the part of the ontology an answer is about ─────────────

_CHAIN_TTL = """
@prefix : <http://olaf.local/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Thing2 a owl:Class ; rdfs:label "Far Away" .
:Vehicle a owl:Class ; rdfs:label "Vehicle" ; rdfs:subClassOf :Thing2 .
:Car a owl:Class ; rdfs:label "Car" ; rdfs:subClassOf :Vehicle .
:Engine a owl:Class ; rdfs:label "Engine" .
:Unrelated a owl:Class ; rdfs:label "Unrelated" .

:hasEngine a owl:ObjectProperty ; rdfs:label "hasEngine" ;
    rdfs:domain :Car ; rdfs:range :Engine .

:TeslaModel3 a owl:NamedIndividual , :Car ; rdfs:label "Tesla Model 3" .
:Car <urn:olaf:extractedFrom> <urn:olaf:chunk:chunk-42> .
"""


def _subjects(ttl: str) -> set[str]:
    from rdflib import Graph

    g = Graph()
    g.parse(data=ttl, format="turtle")
    return {str(s).rsplit("#", 1)[-1] for s in g.subjects()}


def test_extract_subgraph_keeps_focus_and_direct_neighbors_only():
    sub, focus = extract_subgraph(_CHAIN_TTL, ["http://olaf.local/ontology#Car"])
    assert focus == {"http://olaf.local/ontology#Car"}
    subjects = _subjects(sub)
    # Car's parent, the individual typed Car, and the class it links through hasEngine.
    assert {"Car", "Vehicle", "TeslaModel3", "Engine", "hasEngine"} <= subjects
    assert "Unrelated" not in subjects
    # Vehicle's own parent is two hops away: neither drawn nor kept as an edge target.
    assert "Thing2" not in subjects
    assert "Far Away" not in sub


def test_extract_subgraph_without_neighbors_keeps_only_focus_nodes():
    sub, _ = extract_subgraph(_CHAIN_TTL, ["http://olaf.local/ontology#Car"], neighbors=False)
    subjects = _subjects(sub)
    assert "Car" in subjects
    assert "Vehicle" not in subjects
    assert "TeslaModel3" not in subjects


def test_extract_subgraph_drops_uris_that_are_not_ontology_entities():
    _, focus = extract_subgraph(_CHAIN_TTL, [
        "http://www.w3.org/2000/01/rdf-schema#label",
        "urn:olaf:chunk:chunk-42",
        "http://elsewhere.org/X",
        "http://olaf.local/ontology#Engine",
    ])
    assert focus == {"http://olaf.local/ontology#Engine"}


def test_extract_subgraph_brings_in_a_focus_property_domain_and_range():
    sub, focus = extract_subgraph(
        _CHAIN_TTL, ["http://olaf.local/ontology#hasEngine"], neighbors=False,
    )
    assert focus == {"http://olaf.local/ontology#hasEngine"}
    assert {"Car", "Engine", "hasEngine"} <= _subjects(sub)


def test_extract_subgraph_keeps_provenance_for_the_tooltip(tmp_path):
    sub, _ = extract_subgraph(_CHAIN_TTL, ["http://olaf.local/ontology#Car"], neighbors=False)
    out = tmp_path / "sub.html"
    build_pyvis_html(sub, str(out))
    assert "chunk-42" in out.read_text(encoding="utf-8")


def test_extract_subgraph_with_no_focus_is_empty():
    sub, focus = extract_subgraph(_CHAIN_TTL, [])
    assert focus == set()
    assert _subjects(sub) == set()


def test_build_pyvis_html_highlights_focus_nodes(tmp_path):
    out = tmp_path / "hl.html"
    build_pyvis_html(_TTL, str(out), highlight={"http://olaf.local/ontology#Car"})
    content = out.read_text(encoding="utf-8")
    assert HIGHLIGHT_COLOR in content
    assert content.count('"borderWidth": 4') == 1


_INFERRED_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://olaf.local/ontology#> .

ex:Car a owl:Class ; rdfs:label "Car" ; rdfs:subClassOf ex:Vehicle .
ex:Vehicle a owl:Class ; rdfs:label "Vehicle" ; rdfs:subClassOf ex:Artefact .
ex:Artefact a owl:Class ; rdfs:label "Artefact" .
ex:Car rdfs:subClassOf ex:Artefact .
<urn:olaf:stmt:1> a rdf:Statement ; rdf:subject ex:Car ; rdf:predicate rdfs:subClassOf ;
    rdf:object ex:Artefact ; <urn:olaf:inferredBy> "pellet" .
"""


def test_build_pyvis_html_draws_inferred_edges_dotted(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_INFERRED_TTL, str(out))
    html = out.read_text(encoding="utf-8")
    assert html.count('"label": "subClassOf (inferred)"') == 1
    assert html.count('"label": "subClassOf"') == 2


def test_build_pyvis_html_can_hide_inferred_edges(tmp_path):
    out = tmp_path / "graph.html"
    build_pyvis_html(_INFERRED_TTL, str(out), show_inferred=False)
    html = out.read_text(encoding="utf-8")
    assert "(inferred)" not in html
    assert html.count('"label": "subClassOf"') == 2


def test_extract_subgraph_keeps_inferred_marks(tmp_path):
    sub, _ = extract_subgraph(_INFERRED_TTL, ["http://olaf.local/ontology#Car"])
    out = tmp_path / "graph.html"
    build_pyvis_html(sub, str(out))
    assert "subClassOf (inferred)" in out.read_text(encoding="utf-8")
