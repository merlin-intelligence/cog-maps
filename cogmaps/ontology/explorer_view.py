"""Interactive ontology explorer: an exported OLAF Turtle ontology as a self-contained HTML page.

Unlike :func:`cogmaps.ontology.graph.build_pyvis_html` (a bare graph), this page
adds entity/relation search, node and edge filters, clusters by root class, a
1-/2-hop neighborhood focus, side lists of triples/properties/entities and an
inspector panel. It is domain-agnostic: cluster colors come from a generic
palette, datatype values are shown as-is. Colors follow the app's light theme
(:mod:`cogmaps.ui.styles`) so it reads as part of the Ontology Explorer page.

Node classification, inferred triples and provenance are shared with
:mod:`cogmaps.ontology.graph`, so both views draw the same entities.
"""
from __future__ import annotations

import json

from rdflib import OWL, RDF, RDFS, SKOS, Graph, Literal, URIRef

from cogmaps.config import GRAPH_BG_COLOR, GRAPH_EDGE_COLOR, GRAPH_TEXT_COLOR, PROMPT_NODE_COLOR
from cogmaps.ontology.graph import (
    _EXTRACTED_FROM,
    _chunk_id_from_uri,
    _classify,
    _label,
    _local_name,
    inferred_triples,
)

# Pinned, with the same integrity hashes as the build pyvis loads for the other graph views.
_VIS_ROOT = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist"
VIS_NETWORK_URL = f"{_VIS_ROOT}/vis-network.min.js"
_VIS_ASSETS = (
    f'<link rel="stylesheet" href="{_VIS_ROOT}/dist/vis-network.min.css" crossorigin="anonymous" '
    'integrity="sha512-WgxfT5LWjfszlPHXRmBWHkV2eceiWTOBvrKCNbdgDYTHrT2AeLCGbF4sZlZw3UMN3WtL0tGUoIAKsu8mllg/XA==" />\n'
    f'<script src="{VIS_NETWORK_URL}" crossorigin="anonymous" '
    'integrity="sha512-LnvoEWDFrqGHlHmDD2101OrLcbsfkrzoSpvtSQtxK3RMnRV0eOkhhBN2dXHKRrUU8p2DGRTk35n4O8nWSVe1mQ=="></script>'
)

# Root-class clusters get these colors in decreasing size order; the rest share
# CLUSTER_FALLBACK_COLOR. Muted hues that stay readable on the cream background.
CLUSTER_PALETTE = [
    "#4a7fa8", "#2e6b3a", "#9a6010", "#c44a28", "#7a4f9a", "#2f8a8a",
    "#a8456b", "#5b6b2e", "#3f5f9a", "#b07a2a", "#6b3a2e", "#4f7f6a",
]
CLUSTER_FALLBACK_COLOR = "#8a7f74"


def extract_explorer_data(ttl: str, *, title: str = "ontology") -> dict:
    """Parse ``ttl`` into the JSON the explorer page renders.

    Nodes are the ontology's classes and individuals (keyed by full URI, since
    seed and local terms can share a local name), each assigned to the cluster
    of its root class — followed through explicit ``rdfs:subClassOf`` parents,
    or for an individual through its class. Edges: object property assertions
    and schema domain -> range links (``relation``), ``subClassOf``, ``type``
    (individual -> class) and ``disjointWith``, each flagged when the reasoner
    inferred it.
    """
    g = Graph()
    g.parse(data=ttl, format="turtle")
    classes, individuals, object_properties, datatype_properties = _classify(g)
    nodes = classes | individuals
    inferred = inferred_triples(g)

    triple_chunks: dict[tuple, list[str]] = {}
    for st in g.subjects(RDF.type, RDF.Statement):
        chunks = [_chunk_id_from_uri(c) for c in g.objects(st, _EXTRACTED_FROM)]
        if chunks:
            key = (g.value(st, RDF.subject), g.value(st, RDF.predicate), g.value(st, RDF.object))
            triple_chunks.setdefault(key, []).extend(chunks)

    def parents(c) -> list:
        ps = sorted((o for o in g.objects(c, RDFS.subClassOf) if o in classes and o != c), key=str)
        explicit = [p for p in ps if (c, RDFS.subClassOf, p) not in inferred]
        return explicit or ps

    def root_of(c) -> URIRef:
        seen = set()
        while (ps := parents(c)) and c not in seen:
            seen.add(c)
            c = ps[0]
        return c

    def text(n, p) -> str:
        v = g.value(n, p)
        return str(v) if v else ""

    def class_types(i) -> tuple[list, list]:
        explicit, implied = [], []
        for t in sorted(g.objects(i, RDF.type), key=str):
            if t in classes:
                (implied if (i, RDF.type, t) in inferred else explicit).append(t)
        return explicit, implied

    out_nodes = []
    for n in sorted(nodes, key=lambda x: _label(g, x).lower()):
        is_class = n in classes
        node = {
            "id": str(n),
            "label": _label(g, n),
            "kind": "Class" if is_class else "Individual",
            "definition": text(n, SKOS.definition),
            "comment": text(n, RDFS.comment),
            "altLabels": sorted(str(a) for a in g.objects(n, SKOS.altLabel)),
            "chunks": [_chunk_id_from_uri(o) for o in g.objects(n, _EXTRACTED_FROM)],
            "attributes": [
                {
                    "property": _label(g, p),
                    "value": str(o),
                    "datatype": _local_name(o.datatype) if isinstance(o, Literal) and o.datatype else "",
                }
                for p, o in sorted(g.predicate_objects(n), key=str)
                if p in datatype_properties
            ],
        }
        if is_class:
            root = root_of(n)
            node.update(isRoot=not parents(n))
        else:
            explicit, implied = class_types(n)
            node.update(
                isRoot=False,
                types=[str(t) for t in explicit],
                inferredTypes=[str(t) for t in implied],
            )
            primary = (explicit or implied or [None])[0]
            root = root_of(primary) if primary is not None else None
        node["group"] = _label(g, root) if root is not None else "(untyped)"
        out_nodes.append(node)

    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(s, p, o, *, category: str, label: str, schema: bool = False) -> None:
        key = (str(s), str(o), category + label)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({
            "id": f"e{len(edges)}",
            "source": str(s),
            "target": str(o),
            "predicate": str(p),
            "label": label,
            "category": category,
            "inferred": (s, p, o) in inferred,
            "schema": schema,
            "chunks": triple_chunks.get((s, p, o), []),
        })

    usage: dict[URIRef, int] = {}
    for prop in sorted(object_properties, key=str):
        label = _label(g, prop)
        for s, o in g.subject_objects(prop):
            if s in nodes and o in nodes:
                add_edge(s, prop, o, category="relation", label=label)
                usage[prop] = usage.get(prop, 0) + 1
        domain, range_ = g.value(prop, RDFS.domain), g.value(prop, RDFS.range)
        if domain in nodes and range_ in nodes:
            add_edge(domain, prop, range_, category="relation", label=label, schema=True)
    for prop in datatype_properties:
        usage[prop] = sum(1 for s, _ in g.subject_objects(prop) if s in nodes)

    for s, o in sorted(g.subject_objects(RDFS.subClassOf), key=str):
        if s in nodes and o in nodes and s != o:
            add_edge(s, RDFS.subClassOf, o, category="subClassOf", label="subClassOf")
    for i in sorted(individuals, key=str):
        for t in g.objects(i, RDF.type):
            if t in classes:
                add_edge(i, RDF.type, t, category="type", label="type")
    for s, o in sorted(g.subject_objects(OWL.disjointWith), key=str):
        if s in nodes and o in nodes and (str(o), str(s), "disjointWithdisjointWith") not in seen_edges:
            add_edge(s, OWL.disjointWith, o, category="disjointWith", label="disjointWith")

    def labels(xs) -> list[str]:
        return [_label(g, x) for x in xs if isinstance(x, URIRef)]

    properties = [
        {
            "id": str(p),
            "label": _label(g, p),
            "kind": "DatatypeProperty" if p in datatype_properties else "ObjectProperty",
            "domain": labels(g.objects(p, RDFS.domain)),
            "range": labels(g.objects(p, RDFS.range)),
            "usageCount": usage.get(p, 0),
        }
        for p in sorted(object_properties | datatype_properties, key=lambda x: _label(g, x).lower())
    ]

    group_sizes: dict[str, int] = {}
    for n in out_nodes:
        group_sizes[n["group"]] = group_sizes.get(n["group"], 0) + 1
    by_size = sorted(group_sizes, key=lambda k: (-group_sizes[k], k))
    groups = [
        {
            "name": name,
            "count": group_sizes[name],
            "color": CLUSTER_PALETTE[i] if i < len(CLUSTER_PALETTE) else CLUSTER_FALLBACK_COLOR,
        }
        for i, name in enumerate(by_size)
    ]

    relations = [e for e in edges if e["category"] == "relation"]
    stats = {
        "title": title,
        "classCount": len(classes),
        "rootClassCount": sum(1 for n in out_nodes if n["kind"] == "Class" and n["isRoot"]),
        "individualCount": len(individuals),
        "objectPropertyCount": len(object_properties),
        "datatypePropertyCount": len(datatype_properties),
        "relationCount": sum(1 for e in relations if not e["schema"]),
        "schemaRelationCount": sum(1 for e in relations if e["schema"]),
        "attributeCount": sum(len(n["attributes"]) for n in out_nodes),
        "subClassCount": sum(1 for e in edges if e["category"] == "subClassOf" and not e["inferred"]),
        "inferredCount": sum(1 for e in edges if e["inferred"]),
    }
    return {"stats": stats, "nodes": out_nodes, "edges": edges, "properties": properties, "groups": groups}


def build_explorer_html(ttl: str, *, title: str = "ontology") -> str:
    """Render ``ttl`` as the self-contained explorer page (vis-network loaded from a CDN)."""
    data = extract_explorer_data(ttl, title=title)
    # "</" inside the inlined JSON would close the <script> tag early.
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        _TEMPLATE
        .replace("__VIS_ASSETS__", _VIS_ASSETS)
        .replace("__BG__", GRAPH_BG_COLOR)
        .replace("__TEXT__", GRAPH_TEXT_COLOR)
        .replace("__EDGE__", GRAPH_EDGE_COLOR)
        .replace("__ACCENT__", PROMPT_NODE_COLOR)
        .replace("__FALLBACK_COLOR__", CLUSTER_FALLBACK_COLOR)
        .replace("__DATA__", data_json)
    )


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" />
__VIS_ASSETS__
<style>
  :root {
    --bg: __BG__; --card: #ffffff; --hover: #ece6de; --panel: #f2ede6;
    --border: #c0b4a8; --border-soft: #d8d0c6;
    --text: __TEXT__; --muted: __EDGE__; --accent: __ACCENT__; --accent-pale: #c44a2814;
    --green: #2e6b3a; --green-pale: #2e6b3a14; --blue: #4a7fa8; --blue-pale: #4a7fa81a;
    --amber: #9a6010; --amber-pale: #9a60101a; --violet: #7a4f9a; --violet-pale: #7a4f9a1a;
    --mono: 'DM Mono', monospace; --body: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: var(--body); background: var(--bg); color: var(--text); font-size: 12.5px;
    display: flex; flex-direction: column; overflow: hidden;
  }
  .topbar {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    padding: 8px 10px; border-bottom: 1px solid var(--border-soft); background: var(--panel);
  }
  .stats { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-right: auto; }
  .search-box { position: relative; flex: 1 1 220px; max-width: 340px; }
  .search-box input {
    width: 100%; padding: 6px 26px 6px 10px; border: 1px solid var(--border); border-radius: 8px;
    background: var(--card); color: var(--text); font: inherit; outline: none;
  }
  .search-box input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px #c44a2830; }
  .clear-btn {
    position: absolute; right: 6px; top: 50%; transform: translateY(-50%); display: none;
    background: none; border: none; color: var(--muted); cursor: pointer; font-size: 13px;
  }
  .dropdown {
    position: absolute; top: calc(100% + 4px); left: 0; right: 0; z-index: 100; display: none;
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    max-height: 320px; overflow-y: auto; box-shadow: 0 8px 24px rgba(42, 31, 24, 0.15);
  }
  .dropdown-item {
    padding: 6px 10px; border-bottom: 1px solid var(--border-soft); cursor: pointer;
    display: flex; justify-content: space-between; align-items: center; gap: 8px;
  }
  .dropdown-item:hover { background: var(--hover); }
  .sub { font-size: 11px; color: var(--muted); }
  .toolbar {
    display: flex; gap: 6px 14px; align-items: center; flex-wrap: wrap;
    padding: 6px 10px; border-bottom: 1px solid var(--border-soft); font-size: 12px;
  }
  .filter-label {
    font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  }
  label.toggle { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; }
  label.toggle input { accent-color: var(--accent); }
  .spacer { flex: 1; }
  .btn {
    background: transparent; border: 1px solid var(--accent); color: var(--accent); cursor: pointer;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em; padding: 4px 9px; border-radius: 7px;
  }
  .btn:hover, .btn.on { background: var(--accent); color: #fff8f4; }
  select {
    background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 7px;
    padding: 3px 6px; font: inherit; max-width: 200px;
  }
  .workspace { flex: 1; display: flex; overflow: hidden; }
  .side { background: var(--panel); display: flex; flex-direction: column; overflow: hidden; }
  .left { width: 260px; border-right: 1px solid var(--border-soft); }
  .right { width: 300px; border-left: 1px solid var(--border-soft); overflow-y: auto; padding: 12px; }
  .tabs { display: flex; border-bottom: 1px solid var(--border-soft); }
  .tab {
    flex: 1; text-align: center; padding: 7px 4px; cursor: pointer; font-family: var(--mono); font-size: 10.5px;
    color: var(--muted); border-bottom: 2px solid transparent;
  }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); background: var(--bg); }
  .list { flex: 1; overflow-y: auto; padding: 8px; }
  .card {
    background: var(--card); border: 1px solid var(--border-soft); border-radius: 8px;
    padding: 6px 8px; margin-bottom: 6px; cursor: pointer;
  }
  .card:hover, .card.active { border-color: var(--accent); background: #fffaf6; }
  .card-head { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .triple { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; line-height: 1.4; }
  .subj, .obj { font-weight: 600; }
  .obj { color: var(--blue); }
  .lit { color: var(--amber); font-family: var(--mono); font-size: 11px; }
  .pred {
    font-family: var(--mono); font-size: 10.5px; color: var(--green); background: var(--green-pale);
    padding: 1px 5px; border-radius: 4px;
  }
  .pill { font-size: 10px; padding: 1px 7px; border-radius: 999px; font-weight: 600; white-space: nowrap; }
  .pill-class { background: var(--blue-pale); color: var(--blue); }
  .pill-indiv { background: var(--violet-pale); color: var(--violet); }
  .pill-rel { background: var(--green-pale); color: var(--green); }
  .pill-inf { background: var(--amber-pale); color: var(--amber); }
  .pill.link { cursor: pointer; }
  .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  .canvas { flex: 1; position: relative; min-width: 0; }
  #network { width: 100%; height: 100%; }
  .banner {
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index: 10; display: none;
    align-items: center; gap: 8px; padding: 4px 6px 4px 12px; border-radius: 999px;
    background: var(--card); border: 1px solid var(--accent); box-shadow: 0 2px 10px rgba(42, 31, 24, 0.12);
  }
  .legend {
    position: absolute; bottom: 8px; left: 8px; display: flex; flex-wrap: wrap; gap: 4px 12px; pointer-events: none;
    background: rgba(255, 255, 255, 0.85); border: 1px solid var(--border-soft); border-radius: 8px;
    padding: 5px 9px; font-size: 11px; color: var(--muted); max-width: calc(100% - 16px);
  }
  .legend .count { color: var(--text); font-weight: 600; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 2px; word-break: break-word; }
  .uri { font-family: var(--mono); font-size: 10.5px; color: var(--muted); word-break: break-all; margin-bottom: 8px; }
  .section {
    font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
    margin: 12px 0 5px; padding-bottom: 3px; border-bottom: 1px solid var(--border-soft);
  }
  .def {
    background: var(--card); border-left: 3px solid var(--accent); padding: 7px 9px; border-radius: 4px; line-height: 1.5;
  }
  .rel {
    display: flex; justify-content: space-between; align-items: center; gap: 6px; cursor: pointer;
    padding: 4px 7px; margin-bottom: 4px; border-radius: 6px; background: var(--card); border: 1px solid transparent;
  }
  .rel:hover { border-color: var(--accent); }
  .none { color: var(--muted); font-size: 11.5px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .stat { background: var(--card); border: 1px solid var(--border-soft); border-radius: 8px; padding: 6px 8px; }
  .stat b { display: block; font-size: 17px; color: var(--accent); }
  .stat span { font-size: 10.5px; color: var(--muted); }
  .actions { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0 10px; }
  .chunks { font-family: var(--mono); font-size: 10.5px; color: var(--muted); word-break: break-all; }
  .side.collapsed { display: none; }
  .panel-toggle {
    position: absolute; top: 10px; z-index: 11; width: 24px; height: 28px; cursor: pointer;
    background: var(--card); color: var(--muted); border: 1px solid var(--border); border-radius: 6px;
    font-size: 11px; line-height: 1;
  }
  .panel-toggle:hover { color: var(--accent); border-color: var(--accent); }
  #toggleLeft { left: 8px; }
  #toggleRight { right: 8px; }
  @media (max-width: 900px) { .right { width: 240px; } }
</style>
</head>
<body>

<div class="topbar">
  <div class="stats" id="headerStats"></div>
  <div class="search-box">
    <input type="text" id="entitySearch" placeholder="Search a class or individual…" autocomplete="off" />
    <button class="clear-btn" id="clearEntity" title="Clear">✕</button>
    <div class="dropdown" id="entityDropdown"></div>
  </div>
  <div class="search-box">
    <input type="text" id="relationSearch" placeholder="Filter by relation…" autocomplete="off" />
    <button class="clear-btn" id="clearRelation" title="Clear">✕</button>
    <div class="dropdown" id="relationDropdown"></div>
  </div>
</div>

<div class="toolbar">
  <span class="filter-label">nodes</span>
  <label class="toggle"><input type="checkbox" id="chkClasses" checked /> <span id="lblClasses">classes</span></label>
  <label class="toggle"><input type="checkbox" id="chkIndivs" checked /> <span id="lblIndivs">individuals</span></label>
  <span class="filter-label">edges</span>
  <label class="toggle"><input type="checkbox" id="chkRelations" checked /> <span id="lblRelations">relations</span></label>
  <label class="toggle"><input type="checkbox" id="chkSchema" checked /> domain→range</label>
  <label class="toggle"><input type="checkbox" id="chkSubClass" checked /> subClassOf</label>
  <label class="toggle"><input type="checkbox" id="chkType" /> type</label>
  <label class="toggle"><input type="checkbox" id="chkDisjoint" /> disjointWith</label>
  <label class="toggle"><input type="checkbox" id="chkInferred" checked /> inferred</label>
  <span class="spacer"></span>
  <select id="clusterFilter"><option value="ALL">all clusters</option></select>
  <button class="btn" id="btnRelationsOnly" title="Only nodes linked by a relation">relations only</button>
  <button class="btn" id="btnPhysics">resume physics</button>
  <button class="btn" id="btnReset">reset</button>
</div>

<div class="workspace">
  <div class="side left" id="leftPanel">
    <div class="tabs">
      <div class="tab active" data-tab="triples" id="tabTriples">triples</div>
      <div class="tab" data-tab="props" id="tabProps">properties</div>
      <div class="tab" data-tab="entities" id="tabEntities">entities</div>
    </div>
    <div class="list" id="leftList"></div>
  </div>

  <div class="canvas">
    <button class="panel-toggle" id="toggleLeft" data-panel="leftPanel"></button>
    <button class="panel-toggle" id="toggleRight" data-panel="inspector"></button>
    <div class="banner" id="focusBanner">
      <span id="focusText"></span>
      <button class="btn" id="btnExitFocus">full graph</button>
    </div>
    <div id="network"></div>
    <div class="legend">
      <span><span class="swatch" style="background:#ffffff;border:1.5px solid var(--muted)"></span>class</span>
      <span><span class="swatch" style="border-radius:50%;background:var(--muted)"></span>individual</span>
      <span style="color:var(--green)">── relation</span>
      <span>- - subClassOf</span>
      <span>··· inferred</span>
      <span class="count" id="visibleCount"></span>
    </div>
  </div>

  <div class="side right" id="inspector"></div>
</div>

<script>
const DATA = __DATA__;
const S = DATA.stats;
const FALLBACK_COLOR = '__FALLBACK_COLOR__';
const COLORS = { text: '__TEXT__', edge: '__EDGE__', accent: '__ACCENT__', relation: '#2e6b3a',
                 type: '#7a4f9a', disjoint: '#a82020', inferred: '#b0a090' };

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const plural = (n, word, many) => `${n} ${n === 1 ? word : (many || word + 's')}`;

const nodeMap = new Map(DATA.nodes.map(n => [n.id, n]));
const groupColor = new Map(DATA.groups.map(g => [g.name, g.color]));
const nodeLabel = id => (nodeMap.get(id) || {}).label || id;

$('headerStats').textContent = [
  plural(S.classCount, 'class', 'classes'), plural(S.individualCount, 'individual'),
  plural(S.relationCount, 'relation'),
  plural(S.objectPropertyCount + S.datatypePropertyCount, 'property', 'properties'),
].join(' · ');
$('lblClasses').textContent = `classes (${S.classCount})`;
$('lblIndivs').textContent = `individuals (${S.individualCount})`;
$('lblRelations').textContent = `relations (${S.relationCount})`;
$('tabTriples').textContent = `triples (${S.relationCount + S.attributeCount})`;
$('tabProps').textContent = `properties (${DATA.properties.length})`;
$('tabEntities').textContent = `entities (${DATA.nodes.length})`;

DATA.groups.forEach(g => {
  const opt = document.createElement('option');
  opt.value = g.name;
  opt.textContent = `${g.name} (${g.count})`;
  $('clusterFilter').appendChild(opt);
});

// ── graph styling ──
function mix(hex, other, ratio) {
  const a = parseInt(hex.slice(1), 16), b = parseInt(other.slice(1), 16);
  const ch = (x, s) => (x >> s) & 255;
  const m = s => Math.round(ch(a, s) * (1 - ratio) + ch(b, s) * ratio);
  return '#' + ((1 << 24) + (m(16) << 16) + (m(8) << 8) + m(0)).toString(16).slice(1);
}

function tooltip(n) {
  const lines = [`${n.label} (${n.kind === 'Class' ? 'class' : 'individual'}) · ${n.group}`];
  if (n.definition) lines.push(n.definition);
  if (n.comment && n.comment !== n.definition) lines.push(n.comment);
  n.attributes.forEach(a => lines.push(`${a.property}: ${a.value}`));
  return lines.join('\n');
}

function visNode(n) {
  const color = groupColor.get(n.group) || FALLBACK_COLOR;
  if (n.kind === 'Individual') {
    return {
      id: n.id, label: n.label, title: tooltip(n), shape: 'dot', size: 11,
      color: { background: color, border: mix(color, '#000000', 0.25),
               highlight: { background: COLORS.accent, border: COLORS.accent },
               hover: { background: color, border: COLORS.accent } },
      font: { color: COLORS.text, size: 12 },
    };
  }
  return {
    id: n.id, label: n.label, title: tooltip(n), shape: 'box', margin: 7,
    borderWidth: n.isRoot ? 2.5 : 1.2,
    color: { background: mix(color, '#ffffff', 0.82), border: color,
             highlight: { background: '#fff3ec', border: COLORS.accent },
             hover: { background: mix(color, '#ffffff', 0.7), border: color } },
    font: { color: COLORS.text, size: n.isRoot ? 13 : 12 },
  };
}

function visEdge(e) {
  const base = { id: e.id, from: e.source, to: e.target, arrows: 'to' };
  if (e.category === 'relation') {
    return { ...base, label: e.label + (e.inferred ? ' (inferred)' : ''),
             color: { color: e.inferred ? COLORS.inferred : COLORS.relation, highlight: COLORS.accent },
             width: e.schema ? 1.2 : 2, dashes: e.inferred ? [2, 5] : (e.schema ? [8, 3] : false) };
  }
  if (e.category === 'subClassOf') {
    return { ...base, label: e.inferred ? 'subClassOf (inferred)' : '',
             color: { color: e.inferred ? COLORS.inferred : COLORS.edge, highlight: COLORS.accent },
             width: 1.2, dashes: e.inferred ? [2, 5] : [5, 4] };
  }
  if (e.category === 'type') {
    return { ...base, label: 'type', color: { color: e.inferred ? COLORS.inferred : COLORS.type, highlight: COLORS.accent },
             width: 1, dashes: [2, 4] };
  }
  return { ...base, arrows: '', label: 'disjoint',
           color: { color: COLORS.disjoint, highlight: COLORS.accent }, width: 1.1, dashes: [4, 4] };
}

const network = new vis.Network($('network'), { nodes: [], edges: [] }, {
  nodes: { font: { face: 'Inter, sans-serif' }, shadow: false },
  edges: {
    font: { color: COLORS.edge, size: 10, align: 'middle', face: 'DM Mono, monospace',
            background: 'rgba(250,246,240,0.85)', strokeWidth: 0 },
    smooth: { type: 'dynamic' }, selectionWidth: 1.5,
  },
  physics: {
    solver: 'barnesHut',
    barnesHut: { gravitationalConstant: -6000, centralGravity: 0.25, springLength: 130,
                 springConstant: 0.04, damping: 0.12, avoidOverlap: 0.4 },
    stabilization: { iterations: 400, fit: true },
  },
  interaction: { hover: true, tooltipDelay: 150 },
});

// Physics runs only until each new layout settles — left on, vis-network keeps nodes vibrating.
let physicsPinned = false;
network.on('stabilizationIterationsDone', () => {
  if (!physicsPinned) network.setOptions({ physics: { enabled: false } });
});

// ── state & filtering ──
let predicateFilter = null;
let focusedNodeId = null;
let focusHops = 1;
let relationsOnly = false;
let activeTab = 'triples';

function edgeAllowed(e, f) {
  if (e.inferred && !f.inferred) return false;
  if (e.category === 'relation') return f.relations && (!e.schema || f.schema);
  if (e.category === 'subClassOf') return f.subClass;
  if (e.category === 'type') return f.type;
  return f.disjoint;
}

function matchesPredicate(e) {
  if (!predicateFilter) return true;
  const q = predicateFilter.toLowerCase();
  return e.predicate.toLowerCase() === q || e.label.toLowerCase().includes(q);
}

function refreshGraph() {
  const f = {
    classes: $('chkClasses').checked, indivs: $('chkIndivs').checked, relations: $('chkRelations').checked,
    schema: $('chkSchema').checked, subClass: $('chkSubClass').checked, type: $('chkType').checked,
    disjoint: $('chkDisjoint').checked, inferred: $('chkInferred').checked,
  };
  const cluster = $('clusterFilter').value;

  const allowed = DATA.edges.filter(e => edgeAllowed(e, f));
  let edges = allowed.filter(matchesPredicate);

  let neighborhood = null;
  if (focusedNodeId) {
    const allowedIds = new Set(allowed.map(e => e.id));
    neighborhood = new Set([focusedNodeId]);
    for (let hop = 0; hop < focusHops; hop++) {
      const next = new Set(neighborhood);
      // An individual's class is always part of its neighborhood, even with type edges hidden.
      DATA.edges.forEach(e => {
        if (!allowedIds.has(e.id) &&!(hop === 0 && e.category === 'type' && e.source === focusedNodeId)) return;
        if (neighborhood.has(e.source)) next.add(e.target);
        if (neighborhood.has(e.target)) next.add(e.source);
      });
      neighborhood = next;
    }
    edges = edges.filter(e => neighborhood.has(e.source) && neighborhood.has(e.target));
  }

  let endpoints = null;
  if (predicateFilter || relationsOnly) {
    endpoints = new Set();
    edges.forEach(e => {
      if (!relationsOnly || e.category === 'relation') { endpoints.add(e.source); endpoints.add(e.target); }
    });
  }

  const nodes = DATA.nodes.filter(n => {
    if (neighborhood) return neighborhood.has(n.id) && (!endpoints || endpoints.has(n.id) || n.id === focusedNodeId);
    if (endpoints && !endpoints.has(n.id)) return false;
    if (!endpoints && n.kind === 'Class' && !f.classes) return false;
    if (!endpoints && n.kind === 'Individual' && !f.indivs) return false;
    return cluster === 'ALL' || n.group === cluster;
  });
  const ids = new Set(nodes.map(n => n.id));
  edges = edges.filter(e => ids.has(e.source) && ids.has(e.target));

  network.setOptions({ physics: { enabled: true } });
  network.setData({ nodes: new vis.DataSet(nodes.map(visNode)), edges: new vis.DataSet(edges.map(visEdge)) });
  if (focusedNodeId && ids.has(focusedNodeId)) network.selectNodes([focusedNodeId]);
  $('visibleCount').textContent = `${plural(nodes.length, 'node')} · ${plural(edges.length, 'edge')} shown`;
  renderLeftPanel();
}

// ── left panel ──
function card(html, attrs, active) {
  return `<div class="card${active ? ' active' : ''}" ${attrs}>${html}</div>`;
}

function renderLeftPanel() {
  const relQ = $('relationSearch').value.trim().toLowerCase();
  const entQ = $('entitySearch').value.trim().toLowerCase();
  let html = '';

  if (activeTab === 'triples') {
    const triples = DATA.edges.filter(e => e.category === 'relation' && !e.schema).map(e => ({
      source: e.source, label: e.label, object: nodeLabel(e.target), literal: false, inferred: e.inferred,
    }));
    DATA.nodes.forEach(n => n.attributes.forEach(a => triples.push({
      source: n.id, label: a.property, object: a.value, literal: true, inferred: false,
    })));
    const shown = triples.filter(t =>
      (!relQ || t.label.toLowerCase().includes(relQ)) &&
      (!entQ || nodeLabel(t.source).toLowerCase().includes(entQ) || t.object.toLowerCase().includes(entQ)));
    html = shown.map(t => card(`
      <div class="triple">
        <span class="subj">${esc(nodeLabel(t.source))}</span>
        <span class="pred">${esc(t.label)}</span>
        <span class="${t.literal ? 'lit' : 'obj'}">${esc(t.object)}</span>
        ${t.inferred ? '<span class="pill pill-inf">inferred</span>' : ''}
      </div>`, `data-focus="${esc(t.source)}"`)).join('');
    if (!shown.length) html = '<div class="none">No relation or attribute assertion.</div>';
  } else if (activeTab === 'props') {
    const props = DATA.properties
      .filter(p => !relQ || p.label.toLowerCase().includes(relQ) || p.id.toLowerCase().includes(relQ))
      .sort((a, b) => b.usageCount - a.usageCount);
    html = props.map(p => card(`
      <div class="card-head">
        <strong style="color:var(--green)">${esc(p.label)}</strong>
        <span class="pill ${p.usageCount ? 'pill-rel' : 'pill-class'}">${p.usageCount}</span>
      </div>
      <div class="sub">${p.kind === 'ObjectProperty' ? 'object' : 'datatype'} · ${esc(p.domain.join(', ') || 'any')} → ${esc(p.range.join(', ') || 'any')}</div>`,
      `data-pred="${esc(p.id)}" data-kind="${p.kind}"`, predicateFilter === p.id)).join('');
    if (!props.length) html = '<div class="none">No property.</div>';
  } else {
    const ents = DATA.nodes.filter(n => !entQ || n.label.toLowerCase().includes(entQ) ||
      n.id.toLowerCase().includes(entQ) || n.definition.toLowerCase().includes(entQ));
    html = ents.map(n => card(`
      <div class="card-head">
        <strong>${esc(n.label)}</strong>
        <span class="pill ${n.kind === 'Class' ? 'pill-class' : 'pill-indiv'}">${n.kind === 'Class' ? 'class' : 'individual'}</span>
      </div>
      <div class="sub"><span class="swatch" style="background:${groupColor.get(n.group) || FALLBACK_COLOR}"></span>${esc(n.group)}</div>`,
      `data-select="${esc(n.id)}"`)).join('');
    if (!ents.length) html = '<div class="none">No matching entity.</div>';
  }
  $('leftList').innerHTML = html;
}

$('leftList').addEventListener('click', ev => {
  const el = ev.target.closest('.card');
  if (!el) return;
  if (el.dataset.focus) { focusOnNode(el.dataset.focus, 1); inspectNode(el.dataset.focus); }
  else if (el.dataset.select) selectAndInspect(el.dataset.select);
  else if (el.dataset.pred && el.dataset.kind === 'ObjectProperty') {
    setPredicateFilter(predicateFilter === el.dataset.pred ? null : el.dataset.pred);
  }
});

// ── inspector ──
function renderOverview() {
  $('inspector').innerHTML = `
    <h2>${esc(S.title)}</h2>
    <div class="def" style="margin-top:6px">Click a node, a triple or an entity — or search above — to inspect it.
      Double-click a node to isolate its neighborhood.</div>
    <div class="section">summary</div>
    <div class="grid">
      <div class="stat"><b>${S.classCount}</b><span>classes (${S.rootClassCount} roots)</span></div>
      <div class="stat"><b>${S.individualCount}</b><span>individuals</span></div>
      <div class="stat"><b>${S.relationCount}</b><span>relation assertions</span></div>
      <div class="stat"><b>${S.attributeCount}</b><span>attribute values</span></div>
      <div class="stat"><b>${S.subClassCount}</b><span>explicit subClassOf</span></div>
      <div class="stat"><b>${S.inferredCount}</b><span>inferred edges</span></div>
    </div>
    <div class="section">clusters (${DATA.groups.length})</div>
    ${DATA.groups.slice(0, 15).map(g => `
      <div class="rel" data-cluster="${esc(g.name)}">
        <span><span class="swatch" style="background:${g.color}"></span>${esc(g.name)}</span>
        <span class="sub">${g.count}</span>
      </div>`).join('')}
    ${DATA.groups.length > 15 ? `<div class="none">… ${DATA.groups.length - 15} more in the cluster menu</div>` : ''}`;
}

function edgeList(list, dir) {
  if (!list.length) return '<div class="none">none</div>';
  return list.map(e => {
    const other = dir === 'out' ? e.target : e.source;
    const arrow = dir === 'out'
      ? `<span class="pred">${esc(e.label)}</span> →` : `← <span class="pred">${esc(e.label)}</span>`;
    const badges = (e.inferred ? '<span class="pill pill-inf">inferred</span>' : '') +
                   (e.schema ? '<span class="pill pill-class">schema</span>' : '');
    return `<div class="rel" data-select="${esc(other)}"><span>${arrow} <strong>${esc(nodeLabel(other))}</strong></span>${badges}</div>`;
  }).join('');
}

function inspectNode(id) {
  const n = nodeMap.get(id);
  if (!n) return;
  const out = DATA.edges.filter(e => e.source === id);
  const inc = DATA.edges.filter(e => e.target === id);
  const by = (list, cat) => list.filter(e => e.category === cat);
  const color = groupColor.get(n.group) || FALLBACK_COLOR;

  let html = `
    <div class="card-head" style="align-items:flex-start">
      <h2>${esc(n.label)}</h2>
      <span class="pill ${n.kind === 'Class' ? 'pill-class' : 'pill-indiv'}">${n.kind === 'Class' ? 'class' : 'individual'}</span>
    </div>
    <div class="uri">${esc(n.id)}</div>
    <div class="sub" style="margin-bottom:6px"><span class="swatch" style="background:${color}"></span>cluster: ${esc(n.group)}</div>
    <div class="actions">
      <button class="btn" data-hops="1">isolate 1-hop</button>
      <button class="btn" data-hops="2">2-hop</button>
    </div>
    <div class="def">${esc(n.definition || n.comment) || '<i>No definition recorded.</i>'}</div>`;
  if (n.comment && n.definition && n.comment !== n.definition) html += `<div class="def" style="margin-top:6px">${esc(n.comment)}</div>`;
  if (n.altLabels.length) html += `<div class="sub" style="margin-top:6px">also: ${esc(n.altLabels.join(', '))}</div>`;

  if (n.attributes.length) {
    html += `<div class="section">attributes (${n.attributes.length})</div>` + n.attributes.map(a => `
      <div class="rel" style="cursor:default"><span>${esc(a.property)}</span>
        <span class="lit" title="${esc(a.datatype)}">${esc(a.value)}</span></div>`).join('');
  }
  if (n.kind === 'Individual') {
    html += `<div class="section">types</div><div style="display:flex;gap:4px;flex-wrap:wrap">` +
      n.types.map(t => `<span class="pill pill-class link" data-select="${esc(t)}">${esc(nodeLabel(t))}</span>`).join('') +
      n.inferredTypes.map(t => `<span class="pill pill-inf link" data-select="${esc(t)}">${esc(nodeLabel(t))} (inferred)</span>`).join('') +
      (n.types.length + n.inferredTypes.length ? '' : '<span class="none">none</span>') + '</div>';
  }
  const relOut = by(out, 'relation'), relIn = by(inc, 'relation');
  html += `<div class="section">outgoing relations (${relOut.length})</div>${edgeList(relOut, 'out')}`;
  html += `<div class="section">incoming relations (${relIn.length})</div>${edgeList(relIn, 'in')}`;
  if (n.kind === 'Class') {
    const sup = by(out, 'subClassOf'), sub = by(inc, 'subClassOf'), inst = by(inc, 'type');
    html += `<div class="section">superclasses (${sup.length})</div>${edgeList(sup, 'out')}`;
    html += `<div class="section">subclasses (${sub.length})</div>${edgeList(sub, 'in')}`;
    html += `<div class="section">individuals (${inst.length})</div>${edgeList(inst, 'in')}`;
    const disj = DATA.edges.filter(e => e.category === 'disjointWith' && (e.source === id || e.target === id));
    if (disj.length) html += `<div class="section">disjoint with (${disj.length})</div>` +
      disj.map(e => { const o = e.source === id ? e.target : e.source;
        return `<div class="rel" data-select="${esc(o)}"><strong>${esc(nodeLabel(o))}</strong></div>`; }).join('');
  }
  html += `<div class="section">source chunks (${n.chunks.length})</div>
    <div class="chunks">${n.chunks.map(esc).join('<br/>') || 'none'}</div>`;

  $('inspector').innerHTML = html;
  $('inspector').dataset.node = id;
}

$('inspector').addEventListener('click', ev => {
  const el = ev.target.closest('[data-select],[data-hops],[data-cluster]');
  if (!el) return;
  if (el.dataset.select) selectAndInspect(el.dataset.select);
  else if (el.dataset.hops) focusOnNode($('inspector').dataset.node, Number(el.dataset.hops));
  else if (el.dataset.cluster) { $('clusterFilter').value = el.dataset.cluster; exitFocus(); }
});

// ── navigation ──
function centerOn(id) {
  network.selectNodes([id]);
  network.focus(id, { scale: 1.1, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
}

function selectAndInspect(id) {
  inspectNode(id);
  if (network.body.data.nodes.get(id)) centerOn(id);
  else focusOnNode(id, 1);
}

function focusOnNode(id, hops) {
  focusedNodeId = id;
  focusHops = hops || 1;
  $('focusText').innerHTML = `<strong>${esc(nodeLabel(id))}</strong> · ${focusHops}-hop neighborhood`;
  $('focusBanner').style.display = 'flex';
  refreshGraph();
}

function exitFocus() {
  focusedNodeId = null;
  $('focusBanner').style.display = 'none';
  refreshGraph();
}

function setPredicateFilter(pred) {
  predicateFilter = pred;
  $('relationSearch').value = pred ? (DATA.properties.find(p => p.id === pred) || {}).label || pred : '';
  $('clearRelation').style.display = pred ? 'block' : 'none';
  $('relationDropdown').style.display = 'none';
  focusedNodeId = null;
  $('focusBanner').style.display = 'none';
  refreshGraph();
}

// ── search boxes ──
$('entitySearch').addEventListener('input', () => {
  const q = $('entitySearch').value.trim().toLowerCase();
  $('clearEntity').style.display = q ? 'block' : 'none';
  renderLeftPanel();
  const drop = $('entityDropdown');
  if (!q) { drop.style.display = 'none'; return; }
  const matches = DATA.nodes.filter(n => n.label.toLowerCase().includes(q) || n.id.toLowerCase().includes(q) ||
    n.definition.toLowerCase().includes(q)).slice(0, 15);
  drop.innerHTML = matches.length ? matches.map(n => `
    <div class="dropdown-item" data-select="${esc(n.id)}">
      <div><strong>${esc(n.label)}</strong><div class="sub">${esc(n.group)}</div></div>
      <span class="pill ${n.kind === 'Class' ? 'pill-class' : 'pill-indiv'}">${n.kind === 'Class' ? 'class' : 'individual'}</span>
    </div>`).join('') : '<div class="dropdown-item">No matching entity</div>';
  drop.style.display = 'block';
});
$('entityDropdown').addEventListener('click', ev => {
  const el = ev.target.closest('[data-select]');
  if (!el) return;
  $('entityDropdown').style.display = 'none';
  selectAndInspect(el.dataset.select);
});
$('clearEntity').onclick = () => {
  $('entitySearch').value = '';
  $('clearEntity').style.display = 'none';
  $('entityDropdown').style.display = 'none';
  renderLeftPanel();
};

$('relationSearch').addEventListener('input', () => {
  const q = $('relationSearch').value.trim().toLowerCase();
  $('clearRelation').style.display = q ? 'block' : 'none';
  predicateFilter = q || null;
  refreshGraph();
  const drop = $('relationDropdown');
  const matches = q ? DATA.properties.filter(p => p.kind === 'ObjectProperty' &&
    (p.label.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)))
    .sort((a, b) => b.usageCount - a.usageCount).slice(0, 12) : [];
  drop.innerHTML = matches.map(p => `
    <div class="dropdown-item" data-pred="${esc(p.id)}">
      <div><strong style="color:var(--green)">${esc(p.label)}</strong>
        <div class="sub">${esc(p.domain.join(', ') || 'any')} → ${esc(p.range.join(', ') || 'any')}</div></div>
      <span class="pill pill-rel">${p.usageCount}</span>
    </div>`).join('');
  drop.style.display = matches.length ? 'block' : 'none';
});
$('relationDropdown').addEventListener('click', ev => {
  const el = ev.target.closest('[data-pred]');
  if (el) setPredicateFilter(el.dataset.pred);
});
$('clearRelation').onclick = () => setPredicateFilter(null);

document.addEventListener('click', ev => {
  if (!ev.target.closest('.search-box')) {
    $('entityDropdown').style.display = 'none';
    $('relationDropdown').style.display = 'none';
  }
});

// ── controls ──
network.on('click', p => { if (p.nodes.length) inspectNode(p.nodes[0]); });
network.on('doubleClick', p => { if (p.nodes.length) { inspectNode(p.nodes[0]); focusOnNode(p.nodes[0], 1); } });

['chkClasses', 'chkIndivs', 'chkRelations', 'chkSchema', 'chkSubClass', 'chkType', 'chkDisjoint', 'chkInferred',
 'clusterFilter'].forEach(id => $(id).addEventListener('change', refreshGraph));

$('btnRelationsOnly').onclick = function () {
  relationsOnly = !relationsOnly;
  this.classList.toggle('on', relationsOnly);
  exitFocus();
};
$('btnExitFocus').onclick = exitFocus;

$('btnPhysics').onclick = function () {
  physicsPinned = !physicsPinned;
  network.setOptions({ physics: { enabled: physicsPinned } });
  this.textContent = physicsPinned ? 'freeze physics' : 'resume physics';
  this.classList.toggle('on', physicsPinned);
};

$('btnReset').onclick = () => {
  predicateFilter = null;
  relationsOnly = false;
  $('btnRelationsOnly').classList.remove('on');
  ['entitySearch', 'relationSearch'].forEach(id => { $(id).value = ''; });
  ['clearEntity', 'clearRelation'].forEach(id => { $(id).style.display = 'none'; });
  $('clusterFilter').value = 'ALL';
  ['chkClasses', 'chkIndivs', 'chkRelations', 'chkSchema', 'chkSubClass', 'chkInferred']
    .forEach(id => { $(id).checked = true; });
  ['chkType', 'chkDisjoint'].forEach(id => { $(id).checked = false; });
  renderOverview();
  exitFocus();
};

// ── collapsible side panels ──
// Remembered across Streamlit reruns, which rebuild this iframe; storage may be unavailable.
const PANEL_ARROWS = { toggleLeft: ['◀', '▶'], toggleRight: ['▶', '◀'] };
function setPanelOpen(btn, open) {
  $(btn.dataset.panel).classList.toggle('collapsed', !open);
  btn.textContent = PANEL_ARROWS[btn.id][open ? 0 : 1];
  btn.title = `${open ? 'Hide' : 'Show'} the ${btn.id === 'toggleLeft' ? 'lists' : 'inspector'} panel`;
  try { localStorage.setItem(`cogmaps-explorer-${btn.id}`, open ? 'open' : 'closed'); } catch (e) {}
  network.setSize('100%', '100%');
  network.redraw();
}
['toggleLeft', 'toggleRight'].forEach(id => {
  const btn = $(id);
  let saved = null;
  try { saved = localStorage.getItem(`cogmaps-explorer-${id}`); } catch (e) {}
  setPanelOpen(btn, saved ? saved === 'open' : !(id === 'toggleLeft' && window.innerWidth < 900));
  btn.onclick = () => setPanelOpen(btn, $(btn.dataset.panel).classList.contains('collapsed'));
});

document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeTab = tab.dataset.tab;
    renderLeftPanel();
  };
});

renderOverview();
refreshGraph();
</script>
</body>
</html>
"""
