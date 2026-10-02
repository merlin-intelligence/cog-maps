"""Renders an exported OLAF Turtle ontology as a navigable pyvis graph.

Styling mirrors :mod:`cogmaps.graph.explorer` (same background/text/edge
colors, a frozen-after-stabilization physics setup) so the ontology graph
tab reads as part of the same app rather than a foreign embed.
"""
from __future__ import annotations

import html
import json
from collections.abc import Collection, Iterable

from pyvis.network import Network
from rdflib import OWL, RDF, RDFS, SKOS, BNode, Graph, Literal, URIRef

from cogmaps.config import (
    GRAPH_BG_COLOR,
    GRAPH_EDGE_COLOR,
    GRAPH_TEXT_COLOR,
    METHOD_COLORS,
    PROMPT_NODE_COLOR,
)

CLASS_COLOR = METHOD_COLORS["Theta"]
INDIVIDUAL_COLOR = METHOD_COLORS["Hinge"]
# Border of the nodes an Ontology Explorer answer is built on — same accent as
# the prompt node on the Graph Explorer page.
HIGHLIGHT_COLOR = PROMPT_NODE_COLOR

# OLAF's provenance predicate — see olaf.ontology.concept_create/individual_create,
# which (when given source_chunk_id) insert <uri> <urn:olaf:extractedFrom>
# <urn:olaf:chunk:{id}>. Not currently populated for properties/relations, and not
# added again when concept_create/individual_create is called on an already-existing
# URI (dedup/reuse) — only whatever this triple already holds is surfaced here.
_EXTRACTED_FROM = URIRef("urn:olaf:extractedFrom")
_CHUNK_URI_PREFIX = "urn:olaf:chunk:"
# Vocabulary terms (owl:Thing, rdfs:Resource…) are never drawn as nodes.
_BUILTIN_NAMESPACES = (str(OWL), str(RDF), str(RDFS))


def _is_domain_class(node) -> bool:
    return isinstance(node, URIRef) and not str(node).startswith(_BUILTIN_NAMESPACES)


def _chunk_id_from_uri(uri: URIRef) -> str:
    s = str(uri)
    return s[len(_CHUNK_URI_PREFIX):] if s.startswith(_CHUNK_URI_PREFIX) else s

def _physics_options(show_physics_controls: bool) -> dict:
    return {
        "physics": {
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -8000, "centralGravity": 0.25,
                "springLength": 140, "springConstant": 0.04,
                "damping": 0.1, "avoidOverlap": 0.5,
            },
            "stabilization": {"iterations": 1000, "fit": True},
        },
        "edges": {"color": {"color": GRAPH_EDGE_COLOR, "highlight": GRAPH_EDGE_COLOR}, "arrows": "to"},
        "configure": {"enabled": show_physics_controls, "filter": "physics", "showButton": True},
    }


def _remove_blank_card_chrome(path: str) -> None:
    """Strip pyvis's leftover Bootstrap ``.card`` chrome around the graph.

    pyvis's template wraps ``#mynetwork`` in a Bootstrap ``.card`` div and
    leaves empty ``<center><h1></h1></center>`` placeholders for an unset
    title. ``#mynetwork`` itself gets our cream background, but the card and
    the page body don't — any leftover margin/padding around it (card
    padding, empty heading margins, or the embedding iframe simply being
    taller than the rendered content) shows through as a plain white
    rectangle. Matching the body/card background closes that gap instead of
    trying to pixel-match every box model involved.
    """
    style = (
        f"<style>html,body{{margin:0;padding:0;background:{GRAPH_BG_COLOR};}}"
        f".card{{background:{GRAPH_BG_COLOR} !important;border:none !important;}}</style>\n"
    )
    with open(path, encoding="utf-8") as f:
        content = f.read()
    content = content.replace("</head>", style + "</head>", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _freeze_physics_after_stabilization(path: str) -> None:
    """Turn physics off once the layout settles — see cogmaps.graph.explorer for why."""
    snippet = (
        '<script>\n'
        'if (typeof network !== "undefined") {\n'
        '  network.once("stabilizationIterationsDone", function () {\n'
        '    network.setOptions({ physics: false });\n'
        '  });\n'
        '}\n'
        '</script>\n'
    )
    with open(path, encoding="utf-8") as f:
        content = f.read()
    content = content.replace("</body>", snippet + "</body>", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _local_name(uri: URIRef) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _label(g: Graph, node: URIRef) -> str:
    label = g.value(node, RDFS.label)
    return str(label) if label else _local_name(node)


def _classify(g: Graph) -> tuple[set, set, set, set]:
    """``(classes, individuals, object_properties, datatype_properties)`` of ``g``.

    Classes also include external or seed classes that are only referenced
    through ``rdfs:subClassOf`` or an object property's domain/range.
    """
    classes = set(g.subjects(RDF.type, OWL.Class))
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))
    object_properties = set(g.subjects(RDF.type, OWL.ObjectProperty))
    datatype_properties = set(g.subjects(RDF.type, OWL.DatatypeProperty))

    # Include external or seed classes referenced in subClassOf
    for s, o in g.subject_objects(RDFS.subClassOf):
        if _is_domain_class(s):
            classes.add(s)
        if _is_domain_class(o):
            classes.add(o)

    # Include classes referenced in domain/range of object properties
    for prop in object_properties:
        d = g.value(prop, RDFS.domain)
        r = g.value(prop, RDFS.range)
        if _is_domain_class(d):
            classes.add(d)
        if _is_domain_class(r):
            classes.add(r)

    return classes, individuals, object_properties, datatype_properties


def extract_subgraph(ttl: str, uris: Iterable[str], *, neighbors: bool = True) -> tuple[str, set[str]]:
    """Cut the part of an ontology that ``uris`` are about out of its Turtle.

    Returns ``(subgraph_turtle, focus)``: ``focus`` is the subset of ``uris``
    that are classes, individuals or properties of the ontology (anything else
    — vocabulary terms, chunk URIs, URIs from another graph — is dropped).
    The subgraph keeps the focus classes and individuals, the classes a focus
    property links (its domain and range) and, with ``neighbors``, every class
    or individual directly connected to a focus node. Only triples between
    kept nodes survive (plus their labels, comments, types and provenance),
    so :func:`build_pyvis_html` draws no node beyond them.
    """
    g = Graph()
    g.parse(data=ttl, format="turtle")
    classes, individuals, object_properties, datatype_properties = _classify(g)
    nodes = classes | individuals
    properties = object_properties | datatype_properties

    focus = {URIRef(u) for u in uris} & (nodes | properties)
    keep = focus & nodes
    for prop in focus & properties:
        keep.update(n for n in (g.value(prop, RDFS.domain), g.value(prop, RDFS.range)) if n in nodes)
    if neighbors:
        for n in focus & nodes:
            keep.update(o for o in g.objects(n) if o in nodes)
            keep.update(s for s in g.subjects(object=n) if s in nodes)
            # Schema relations: the class at the other end of a property n is the domain or range of.
            for prop in object_properties:
                ends = (g.value(prop, RDFS.domain), g.value(prop, RDFS.range))
                if n in ends:
                    keep.update(e for e in ends if e in nodes)

    def keeps(s, p, o) -> bool:
        return isinstance(o, Literal) or o in keep or p in (RDF.type, _EXTRACTED_FROM)

    sub = Graph()
    for prefix, ns in g.namespaces():
        sub.bind(prefix, ns)
    used_properties = set(focus & properties)
    for s, p, o in g:
        if s in keep and keeps(s, p, o):
            sub.add((s, p, o))
            if p in properties:
                used_properties.add(p)
    # Declarations of the properties drawn as edges — instance assertions above,
    # or a schema domain -> range edge between two kept classes.
    for prop in properties:
        if prop in used_properties or (g.value(prop, RDFS.domain) in keep and g.value(prop, RDFS.range) in keep):
            for p, o in g.predicate_objects(prop):
                if keeps(prop, p, o):
                    sub.add((prop, p, o))
    return sub.serialize(format="turtle"), {str(u) for u in focus}


def build_pyvis_html(
    ttl: str,
    output_path: str,
    *,
    show_physics_controls: bool = True,
    highlight: Collection[str] = (),
) -> str:
    """Parse ``ttl`` and render a navigable pyvis graph to ``output_path``.

    Nodes: ``owl:Class`` and ``owl:NamedIndividual`` subjects, labeled via
    ``rdfs:label`` (falling back to the URI's local name). Edges:
    ``rdfs:subClassOf`` (hierarchy) and any user-declared ``owl:ObjectProperty``
    relation between two such nodes. Datatype property values, comments, and
    any ``urn:olaf:extractedFrom`` source-chunk link(s) are folded into the
    subject node's tooltip instead of becoming separate nodes.

    ``show_physics_controls`` toggles vis-network's built-in physics-tuning
    panel (sliders for gravity/spring length/etc.) below the graph — on by
    default, off for the Ontology Explorer page.

    Nodes whose URI is in ``highlight`` get a thick :data:`HIGHLIGHT_COLOR`
    border and a larger size.
    """
    g = Graph()
    g.parse(data=ttl, format="turtle")
    classes, individuals, object_properties, datatype_properties = _classify(g)
    nodes = classes | individuals

    # Fold definitions, datatype property values, comments, and source chunk(s) into each
    # node's tooltip.
    tooltips: dict[URIRef, list[str]] = {n: [] for n in nodes}
    for n in nodes:
        comment = g.value(n, RDFS.comment)
        if comment:
            tooltips[n].append(str(comment))
        definition = g.value(n, SKOS.definition)
        if definition:
            tooltips[n].append(str(definition))
        alt_labels = list(g.objects(n, SKOS.altLabel))
        if alt_labels:
            tooltips[n].append(f"alt: {', '.join(str(l) for l in alt_labels)}")
        for p, o in g.predicate_objects(n):
            if p in datatype_properties:
                tooltips[n].append(f"{_label(g, p)}: {o}")
        source_chunks = [_chunk_id_from_uri(o) for o in g.objects(n, _EXTRACTED_FROM)]
        if source_chunks:
            tooltips[n].append(f"source chunk(s): {', '.join(source_chunks)}")

    net = Network(height="800px", width="100%", notebook=False, directed=True,
                  bgcolor=GRAPH_BG_COLOR, font_color=GRAPH_TEXT_COLOR)

    for n in nodes:
        color = CLASS_COLOR if n in classes else INDIVIDUAL_COLOR
        title = html.escape("\n".join(tooltips[n]) or _label(g, n))
        style = (
            {"color": {"background": color, "border": HIGHLIGHT_COLOR}, "borderWidth": 4, "size": 18}
            if str(n) in highlight else {"color": color}
        )
        net.add_node(str(n), label=_label(g, n), title=title,
                     shape="ellipse" if n in classes else "dot", **style)

    edges_added: set[tuple[str, str, str]] = set()

    for s, o in g.subject_objects(RDFS.subClassOf):
        if s in nodes and isinstance(o, URIRef) and o in nodes:
            key = (str(s), str(o), "subClassOf")
            if key not in edges_added:
                net.add_edge(str(s), str(o), label="subClassOf", dashes=True)
                edges_added.add(key)

    for prop in object_properties:
        prop_label = _label(g, prop)
        # 1. Instance assertions (s prop o)
        for s, o in g.subject_objects(prop):
            if isinstance(o, BNode) or s not in nodes or o not in nodes:
                continue
            key = (str(s), str(o), prop_label)
            if key not in edges_added:
                net.add_edge(str(s), str(o), label=prop_label)
                edges_added.add(key)
        # 2. Schema domain -> range relations
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)
        if isinstance(domain, URIRef) and isinstance(range_, URIRef):
            if domain in nodes and range_ in nodes:
                key = (str(domain), str(range_), prop_label)
                if key not in edges_added:
                    net.add_edge(str(domain), str(range_), label=prop_label)
                    edges_added.add(key)

    net.set_options(json.dumps(_physics_options(show_physics_controls)))
    net.save_graph(output_path)
    _freeze_physics_after_stabilization(output_path)
    _remove_blank_card_chrome(output_path)
    return output_path
