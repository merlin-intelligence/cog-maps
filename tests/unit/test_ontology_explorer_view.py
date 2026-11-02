"""Unit tests for cogmaps.ontology.explorer_view: Turtle -> interactive explorer page."""
from __future__ import annotations

import json
import re

from cogmaps.config import GRAPH_BG_COLOR
from cogmaps.ontology.explorer_view import (
    CLUSTER_PALETTE,
    VIS_NETWORK_URL,
    build_explorer_html,
    extract_explorer_data,
)

_NS = "http://olaf.local/ontology#"

_TTL = """
@prefix : <http://olaf.local/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Vehicle a owl:Class ; rdfs:label "Vehicle" .
:Car a owl:Class ; rdfs:label "Car" ; rdfs:subClassOf :Vehicle ; skos:definition "A road vehicle." .
:Engine a owl:Class ; rdfs:label "Engine" .
:Vehicle owl:disjointWith :Engine .

:hasEngine a owl:ObjectProperty ; rdfs:label "has engine" ; rdfs:domain :Car ; rdfs:range :Engine .
:horsepower a owl:DatatypeProperty ; rdfs:label "horsepower" .

:Model3 a owl:NamedIndividual, :Car ; rdfs:label "Model 3" ; :hasEngine :E1 ;
    :horsepower "283"^^xsd:integer .
:E1 a owl:NamedIndividual, :Engine ; rdfs:label "E1" .
:E1 a :Vehicle .

:Car <urn:olaf:extractedFrom> <urn:olaf:chunk:chunk-42> .

:st1 a rdf:Statement ; rdf:subject :E1 ; rdf:predicate rdf:type ; rdf:object :Vehicle ;
    <urn:olaf:inferredBy> "pellet" .
"""


def _node(data, local):
    return next(n for n in data["nodes"] if n["id"] == _NS + local)


def test_nodes_are_clustered_by_root_class():
    data = extract_explorer_data(_TTL)
    assert _node(data, "Car")["group"] == "Vehicle"
    assert _node(data, "Vehicle")["isRoot"] is True
    assert _node(data, "Car")["isRoot"] is False
    # An individual joins the cluster of its explicit (not inferred) class.
    assert _node(data, "Model3")["group"] == "Vehicle"
    assert _node(data, "E1")["group"] == "Engine"


def test_individual_types_split_explicit_and_inferred():
    e1 = _node(extract_explorer_data(_TTL), "E1")
    assert e1["types"] == [_NS + "Engine"]
    assert e1["inferredTypes"] == [_NS + "Vehicle"]


def test_datatype_values_are_kept_raw_with_their_datatype():
    attrs = _node(extract_explorer_data(_TTL), "Model3")["attributes"]
    assert attrs == [{"property": "horsepower", "value": "283", "datatype": "integer"}]


def test_definition_and_source_chunks_are_surfaced():
    car = _node(extract_explorer_data(_TTL), "Car")
    assert car["definition"] == "A road vehicle."
    assert car["chunks"] == ["chunk-42"]


def test_edges_cover_every_category():
    edges = extract_explorer_data(_TTL)["edges"]
    by_cat = {}
    for e in edges:
        by_cat.setdefault(e["category"], []).append(e)
    assertion = [e for e in by_cat["relation"] if not e["schema"]]
    schema = [e for e in by_cat["relation"] if e["schema"]]
    assert [(e["source"], e["target"], e["label"]) for e in assertion] == [(_NS + "Model3", _NS + "E1", "has engine")]
    assert [(e["source"], e["target"]) for e in schema] == [(_NS + "Car", _NS + "Engine")]
    assert [(e["source"], e["target"]) for e in by_cat["subClassOf"]] == [(_NS + "Car", _NS + "Vehicle")]
    assert len(by_cat["disjointWith"]) == 1
    inferred_type = [e for e in by_cat["type"] if e["inferred"]]
    assert [(e["source"], e["target"]) for e in inferred_type] == [(_NS + "E1", _NS + "Vehicle")]


def test_clusters_get_palette_colors_by_size():
    groups = extract_explorer_data(_TTL)["groups"]
    assert groups[0] == {"name": "Vehicle", "count": 3, "color": CLUSTER_PALETTE[0]}
    assert groups[1]["name"] == "Engine"


def test_stats_count_assertions_not_schema_links():
    stats = extract_explorer_data(_TTL, title="cars")["stats"]
    assert stats["title"] == "cars"
    assert stats["classCount"] == 3
    assert stats["individualCount"] == 2
    assert stats["relationCount"] == 1
    assert stats["schemaRelationCount"] == 1
    assert stats["attributeCount"] == 1


def test_html_embeds_data_light_theme_and_pinned_vis_network():
    page = build_explorer_html(_TTL, title="cars")
    assert VIS_NETWORK_URL in page and "9.1.2" in VIS_NETWORK_URL
    assert GRAPH_BG_COLOR in page
    assert not re.search(r"__[A-Z_]+__", page)
    data = json.loads(re.search(r"const DATA = (.*);\n", page).group(1))
    assert data["stats"]["title"] == "cars"


def test_html_cannot_be_broken_out_of_by_a_label():
    ttl = _TTL.replace('rdfs:label "Engine"', 'rdfs:label "</script><script>alert(1)</script>"')
    page = build_explorer_html(ttl)
    assert "</script><script>alert(1)" not in page


def test_empty_ontology_renders():
    data = extract_explorer_data("")
    assert data["nodes"] == [] and data["edges"] == [] and data["groups"] == []
    assert "const DATA" in build_explorer_html("")
