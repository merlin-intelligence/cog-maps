"""Ontology Explorer page — browse ontologies stored in Oxigraph, run read-only SPARQL."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from cogmaps.config import TEMP_GRAPH_OUTPUTS, oxigraph_url
from cogmaps.ontology.graph import build_pyvis_html
from cogmaps.ontology.oxigraph_client import (
    SparqlError,
    export_graph_ttl,
    is_reachable,
    list_graphs,
    run_query,
)
from cogmaps.ui.auth import check_password
from cogmaps.ui.components import empty_state, render_sidebar, section_header
from cogmaps.ui.styles import apply_global_styles

apply_global_styles()
if not check_password():
    st.stop()
render_sidebar()

section_header("/explore ontology/", "browse your knowledge model")

_OXIGRAPH_URL = oxigraph_url()

if not is_reachable(_OXIGRAPH_URL):
    empty_state(
        "⚡",
        f"Oxigraph is offline at <code>{_OXIGRAPH_URL}</code>. "
        "Start it with <code>docker-compose up -d</code> and refresh.",
    )
    st.stop()

try:
    graphs = list_graphs(_OXIGRAPH_URL)
except SparqlError as e:
    st.error(f"Could not list ontologies: {e}")
    st.stop()

if not graphs:
    empty_state("🧬", "No ontology found in Oxigraph yet. Go to <strong>/build ontology/</strong> to build one first.")
    st.stop()


def _graph_label(uri: str) -> str:
    return uri.removeprefix("urn:olaf:")


graph_map = {_graph_label(g): g for g in graphs}
graph_label = st.selectbox("ontology (named graph)", sorted(graph_map))
graph_uri = graph_map[graph_label]

tab_graph, tab_sparql = st.tabs(["🕸 graph view", "🔎 SPARQL query"])

with tab_graph:
    try:
        ttl = export_graph_ttl(_OXIGRAPH_URL, graph_uri)
    except SparqlError as e:
        st.error(f"Could not export this ontology: {e}")
    else:
        if not ttl.strip():
            empty_state("🧬", "This ontology graph is empty.")
        else:
            graph_dir = os.path.join(str(TEMP_GRAPH_OUTPUTS), "ontology_explorer")
            os.makedirs(graph_dir, exist_ok=True)
            safe_name = graph_label.replace(":", "_").replace("/", "_")
            html_path = os.path.join(graph_dir, f"{safe_name}.html")
            try:
                build_pyvis_html(ttl, html_path, show_physics_controls=False)
                with open(html_path, encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=850, scrolling=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not render the graph view: {e}")

            with st.expander("raw Turtle"):
                st.code(ttl, language="turtle")
                st.download_button(
                    "download .ttl", ttl.encode("utf-8"),
                    file_name=f"{safe_name}.ttl", mime="text/turtle",
                )

with tab_sparql:
    st.caption("Read-only SPARQL against Oxigraph — SELECT, ASK, CONSTRUCT, or DESCRIBE.")
    default_query = f"SELECT ?s ?p ?o WHERE {{\n  GRAPH <{graph_uri}> {{ ?s ?p ?o }}\n}}\nLIMIT 50"
    query = st.text_area("SPARQL query", value=default_query, height=180)

    if st.button("▶ run query", type="primary"):
        try:
            kind, payload = run_query(_OXIGRAPH_URL, query)
        except SparqlError as e:
            st.error(str(e))
        else:
            if kind == "table":
                if payload.get("boolean") is not None:
                    st.write(f"**{payload['boolean']}**")
                else:
                    bindings = payload.get("results", {}).get("bindings", [])
                    variables = payload.get("head", {}).get("vars", [])
                    if not bindings:
                        st.info("No results.")
                    else:
                        rows = [
                            {v: b.get(v, {}).get("value", "") for v in variables}
                            for b in bindings
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.code(payload, language="turtle")
