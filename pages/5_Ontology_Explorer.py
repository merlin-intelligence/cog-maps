"""Ontology Explorer page — ask questions over an ontology, browse it, run read-only SPARQL."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import pandas as pd
import streamlit as st

from cogmaps.config import ONTOLOGY_TOOLCALL_MODELS, TEMP_GRAPH_OUTPUTS, oxigraph_url
from cogmaps.core.llm_impacts import record_llm_impacts
from cogmaps.ontology.graph import build_pyvis_html, extract_subgraph
from cogmaps.ontology.olaf_config import sanitize_ontology_id
from cogmaps.ontology.oxigraph_client import (
    SparqlError,
    export_graph_ttl,
    is_reachable,
    list_graphs,
    run_query,
)
from cogmaps.ontology.search_agent import SearchConversation, SearchTurn, ask_collection
from cogmaps.qdrant.store import QdrantStore
from cogmaps.ui.auth import check_password, list_visible_collections
from cogmaps.ui.components import (
    empty_state,
    render_llm_impacts_footer,
    render_sidebar,
    section_header,
    session_llm_calls,
)
from cogmaps.ui.styles import apply_global_styles

logger = logging.getLogger(__name__)

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/explore ontology/", "browse your knowledge model")

_OXIGRAPH_URL = oxigraph_url()

if not sb.is_connected:
    empty_state("⚡", "Qdrant is offline. Start it with <code>docker-compose up -d</code> and refresh.")
    st.stop()

if not is_reachable(_OXIGRAPH_URL):
    empty_state(
        "⚡",
        f"Oxigraph is offline at <code>{_OXIGRAPH_URL}</code>. "
        "Start it with <code>docker-compose up -d</code> and refresh.",
    )
    st.stop()

try:
    graphs = set(list_graphs(_OXIGRAPH_URL))
except SparqlError as e:
    st.error(f"Could not list ontologies: {e}")
    st.stop()

# Ontologies are listed through the collections this user can see, not every
# named graph in Oxigraph — and only those that have been built (the search
# agent's OLAF would otherwise bootstrap an empty ontology on startup).
store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_with_ontology = [
    (label, qname) for label, qname in list_visible_collections(store.list_collections())
    if f"urn:olaf:{sanitize_ontology_id(qname)}" in graphs
]
if not _with_ontology:
    empty_state("🧬", "No ontology found yet. Go to <strong>/build ontology/</strong> to build one first.")
    st.stop()

col_map = dict(_with_ontology)
collection_name = st.selectbox("collection", list(col_map))
qdrant_col = col_map[collection_name]
ontology_id = sanitize_ontology_id(qdrant_col)
graph_uri = f"urn:olaf:{ontology_id}"


def _render_pyvis(ttl: str, *, highlight: set[str] = frozenset()) -> str:
    """Render ``ttl`` with :func:`build_pyvis_html` and return the HTML."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "graph.html")
        build_pyvis_html(ttl, path, show_physics_controls=False, highlight=highlight)
        with open(path, encoding="utf-8") as f:
            return f.read()


def _turn_view(turn: SearchTurn) -> dict:
    """Everything the page shows under an answer, computed once when it arrives."""
    view: dict = {"focus": set(), "html": None, "html_focus_only": None, "chunks": []}
    try:
        ttl = export_graph_ttl(_OXIGRAPH_URL, graph_uri)
        uris = turn.trace.touched_uris
        sub_ttl, view["focus"] = extract_subgraph(ttl, uris)
        if view["focus"]:
            focus_ttl, _ = extract_subgraph(ttl, uris, neighbors=False)
            view["html"] = _render_pyvis(sub_ttl, highlight=view["focus"])
            view["html_focus_only"] = _render_pyvis(focus_ttl, highlight=view["focus"])
    except Exception as e:  # noqa: BLE001
        view["graph_error"] = str(e)
        logger.exception("Could not build the answer's subgraph")

    # Chunks cited in the answer first, then the other chunks the agent read.
    read = turn.trace.chunks_read
    cited = turn.cited_chunk_ids
    missing = [cid for cid in cited if cid not in read]
    fetched: dict[str, dict] = {}
    if missing:
        try:
            for p in store.client.retrieve(qdrant_col, ids=missing, with_payload=True, with_vectors=False):
                payload = p.payload or {}
                fetched[str(p.id)] = {
                    "id": str(p.id), "doc_id": payload.get("filename", ""),
                    "chunk_index": payload.get("chunk_number", 0), "text": payload.get("text", ""),
                }
        except Exception:  # noqa: BLE001 — a cited id the model mistyped, or Qdrant hiccup
            logger.warning("Could not fetch cited chunks %s", missing, exc_info=True)
    view["chunks"] = [
        {**(read.get(cid) or fetched[cid]), "cited": True}
        for cid in cited if cid in read or cid in fetched
    ] + [{**c, "cited": False} for cid, c in read.items() if cid not in cited]
    return view


def _render_evidence(turn: SearchTurn, view: dict, key: str) -> None:
    tab_sub, tab_chunks, tab_trace = st.tabs(["🕸 ontology", "📋 chunks", "🔎 trace"])

    with tab_sub:
        if view.get("graph_error"):
            st.error(f"Could not build the subgraph: {view['graph_error']}")
        elif not view["focus"]:
            st.info("The agent did not use any entity of this ontology for this answer.")
        else:
            show_neighbors = st.checkbox("show direct neighbors", value=True, key=f"{key}_neighbors")
            st.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
                f'{len(view["focus"])} entities used by the agent (thick border)'
                f'{" + their direct neighbors" if show_neighbors else ""}</p>',
                unsafe_allow_html=True,
            )
            st.components.v1.html(
                view["html"] if show_neighbors else view["html_focus_only"], height=650, scrolling=True,
            )

    with tab_chunks:
        if not view["chunks"]:
            st.info("The agent did not read any source chunk for this answer.")
        for i, ch in enumerate(view["chunks"]):
            label = f"{i + 1}. {ch.get('doc_id', '')} · chunk {ch.get('chunk_index', '')}"
            with st.expander(label + (" · cited" if ch["cited"] else "")):
                st.caption(f"chunk id: {ch['id']}")
                st.write(ch.get("text", ""))

    with tab_trace:
        if turn.trace.sparql_queries:
            st.markdown("**SPARQL queries**")
            for q in turn.trace.sparql_queries:
                st.code(q["query"], language="sparql")
                if not q["ok"]:
                    st.caption(f"✗ {q['result']}")
        st.markdown(f"**Tool calls** ({len(turn.trace.tool_calls)})")
        st.code(
            "\n".join(
                f"{'✓' if c.ok else '✗'} {c.name}({', '.join(f'{k}={v!r}' for k, v in c.args.items())})"
                for c in turn.trace.tool_calls
            ) or "(none)",
            language="text",
        )


tab_ask, tab_browse, tab_sparql = st.tabs(["❓ ask", "🕸 browse", "🔎 SPARQL query"])

with tab_ask:
    # One conversation per ontology, so switching collection doesn't mix histories.
    conversations = st.session_state.setdefault("ontology_search", {})
    state = conversations.get(ontology_id)
    if state is None:
        state = conversations[ontology_id] = {"conversation": SearchConversation(ontology_id), "views": []}
    conversation: SearchConversation = state["conversation"]

    col_m, col_r = st.columns([3, 1])
    with col_m:
        model = st.selectbox("model (tool-calling)", ONTOLOGY_TOOLCALL_MODELS)
    with col_r:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if st.button("↺ new conversation", disabled=not conversation.turns, use_container_width=True):
            conversation.reset()
            state["views"] = []
            st.rerun()

    if not conversation.turns:
        st.caption(
            "Ask a question: the agent searches the ontology (concepts, relations, SPARQL) and reads "
            "the source chunks behind it. Follow-up questions keep the conversation's context."
        )

    for i, (turn, view) in enumerate(zip(conversation.turns, state["views"])):
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer or "_(empty answer)_")
            with st.expander("evidence", expanded=i == len(conversation.turns) - 1):
                _render_evidence(turn, view, key=f"{ontology_id}_{i}")

    question = st.chat_input("ask a question about this ontology…", key=f"ask_{ontology_id}")
    if question:
        if not sb.scaleway_api_key:
            st.warning("Set a Scaleway API key in the sidebar to use the search agent.")
        else:
            with st.chat_message("user"):
                st.write(question)
            with st.status("searching the ontology…", expanded=True) as status:
                try:
                    with record_llm_impacts(session_llm_calls("ontology_explorer").append):
                        turn = asyncio.run(ask_collection(
                            conversation, question,
                            qdrant_url=f"http://{sb.qdrant_host}:{sb.qdrant_port}",
                            collection=qdrant_col, model=model, api_key=sb.scaleway_api_key,
                            log=status.write,
                        ))
                    status.write("Building the answer's subgraph…")
                    state["views"].append(_turn_view(turn))
                except Exception as e:  # noqa: BLE001
                    status.update(label="search failed", state="error")
                    st.error(f"Search failed: {e}")
                    logger.exception("Ontology search failed")
                else:
                    status.update(label="done", state="complete")
                    st.rerun()

with tab_browse:
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
            html_path = os.path.join(graph_dir, f"{ontology_id}.html")
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
                    file_name=f"{ontology_id}.ttl", mime="text/turtle",
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

render_llm_impacts_footer(session_llm_calls("ontology_explorer"), key="ontology_explorer")
