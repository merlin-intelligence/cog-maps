"""Ontology Building page — OLAF (MCP) + Oxigraph, driven by a tool-calling LLM agent."""
from __future__ import annotations

import os
import uuid

import streamlit as st

from cogmaps.config import ONTOLOGY_TOOLCALL_MODELS, TEMP_GRAPH_OUTPUTS
from cogmaps.ontology.graph import build_pyvis_html
from cogmaps.ontology.olaf_config import sanitize_ontology_id
from cogmaps.ontology.runner import OntologyJobRunner, ttl_path_for
from cogmaps.ontology.store import OntologyJobStore
from cogmaps.qdrant.store import QdrantStore
from cogmaps.ui.auth import check_password, is_admin, list_visible_collections
from cogmaps.ui.components import empty_state, render_sidebar, section_header
from cogmaps.ui.styles import apply_global_styles

# ── Module-level singletons (one per server process, shared across sessions) ──

@st.cache_resource
def _ontology_job_store() -> OntologyJobStore:
    os.makedirs("user_data", exist_ok=True)
    return OntologyJobStore(os.path.join("user_data", "jobs.db"))


@st.cache_resource
def _ontology_job_runner(qdrant_host: str, qdrant_port: int) -> OntologyJobRunner:
    return OntologyJobRunner(_ontology_job_store(), qdrant_host, qdrant_port)


# ── Page setup ────────────────────────────────────────────────────────

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/build ontology/", "construct your knowledge model")

if not sb.is_connected:
    empty_state("⚡", "Qdrant is offline. Start it with <code>docker-compose up -d</code> and refresh.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
# Building an ontology writes into the shared Oxigraph store, same write
# permission model as ingestion: non-admins can only act on their own
# private collections.
_editable = _visible if is_admin() else [
    (label, qname) for label, qname in _visible if not label.startswith("[public] ")
]

if not _editable:
    empty_state("📂", "No collections found. Go to <strong>/enrich corpus/</strong> to ingest documents first.")
    st.stop()

col_labels = [label for label, _ in _editable]
col_map = dict(_editable)

collection_name = st.selectbox("collection", col_labels)
qdrant_col = col_map[collection_name]
ontology_id = sanitize_ontology_id(qdrant_col)
st.caption(f"ontology id: `{ontology_id}` — repeated builds on this collection extend the same ontology")

filenames = sorted(store.existing_filenames(qdrant_col))

# ── Job helpers ───────────────────────────────────────────────────────

def _activate_job(job_id: str) -> None:
    st.session_state["ontology_active_job_id"] = job_id
    st.session_state.pop("ontology_log_cursor", None)
    st.session_state.pop("ontology_log_buffer", None)


@st.fragment(run_every=2.0)
def _render_job_status() -> None:
    job_id = st.session_state.get("ontology_active_job_id")
    if not job_id:
        return
    job = _ontology_job_store().get_job(job_id)
    if not job:
        return

    status = job["status"]
    icons = {"pending": "⏳", "running": "⚙️", "done": "✅", "failed": "❌"}
    st.markdown(f"**{icons.get(status, '·')} ontology build** — `{status}`")

    cur, tot = job["progress_current"], job["progress_total"]
    if tot > 0:
        st.progress(min(cur / tot, 1.0))
        st.caption(f"iteration {cur} / {tot}")
    elif status == "running":
        st.progress(0)

    last_rowid = st.session_state.get("ontology_log_cursor", 0)
    new_entries = _ontology_job_store().get_logs(job_id, after_rowid=last_rowid)
    if new_entries:
        rowids, messages = zip(*new_entries, strict=True)
        st.session_state["ontology_log_cursor"] = rowids[-1]
        buf = st.session_state.get("ontology_log_buffer", "")
        st.session_state["ontology_log_buffer"] = buf + "\n".join(messages) + "\n"

    buf = st.session_state.get("ontology_log_buffer", "")
    if buf:
        with st.expander("build log", expanded=(status == "running")):
            st.code(buf.strip(), language=None)

    if status == "done":
        st.success("Ontology build complete!")
        st.button("↻ view results", on_click=st.rerun)
    elif status == "failed":
        st.error("Ontology build failed — see log above.")


# ── Document selection & launch ─────────────────────────────────────────

if not filenames:
    empty_state("📄", "No documents in this collection yet.")
else:
    selected_docs = st.multiselect(
        "documents to include in this build",
        filenames,
        default=filenames,
        help="Deselect any document to exclude it from this build. Chunks already "
             "processed in a previous build are skipped automatically.",
    )
    model = st.selectbox("model (tool-calling)", ONTOLOGY_TOOLCALL_MODELS)

    if not sb.nebius_api_key:
        st.warning("Set NEBIUS_API_KEY to run the ontology-building agent (Nebius tool-calling only).")
    elif st.button("▶ launch build", type="primary"):
        if not selected_docs:
            st.warning("Select at least one document.")
        else:
            job_id = str(uuid.uuid4())
            _ontology_job_store().create_job(
                job_id=job_id,
                collection=qdrant_col,
                ontology_id=ontology_id,
                doc_filenames=selected_docs,
                llm_provider="nebius",
                llm_model=model,
            )
            _ontology_job_runner(sb.qdrant_host, sb.qdrant_port).submit(job_id)
            _activate_job(job_id)
            st.rerun()

# Reconnect banner: a build already running for this collection but this
# session has no active job (e.g. reconnected after a tab closure).
if "ontology_active_job_id" not in st.session_state:
    latest = _ontology_job_store().latest_job_for(qdrant_col)
    if latest and latest["status"] in ("running", "pending"):
        st.info(f"A build is already running for **{collection_name}** (started {latest['created_at'][:16]}).")
        if st.button("monitor this build"):
            _activate_job(latest["id"])
            st.rerun()

_render_job_status()

st.markdown("---")

# ── Results: raw Turtle + navigable graph ───────────────────────────────

ttl_file = ttl_path_for(qdrant_col)
if os.path.exists(ttl_file):
    with open(ttl_file, encoding="utf-8") as f:
        ttl = f.read()

    tab_raw, tab_graph = st.tabs(["🧬 raw RDF/OWL", "🕸 graph view"])

    with tab_raw:
        st.code(ttl, language="turtle")
        st.download_button(
            "download .ttl", ttl.encode("utf-8"),
            file_name=f"{ontology_id}.ttl", mime="text/turtle",
        )

    with tab_graph:
        graph_dir = os.path.join(str(TEMP_GRAPH_OUTPUTS), "ontology")
        os.makedirs(graph_dir, exist_ok=True)
        graph_html_path = os.path.join(graph_dir, f"{ontology_id}.html")
        try:
            build_pyvis_html(ttl, graph_html_path)
            with open(graph_html_path, encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=850, scrolling=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not render the graph view: {e}")
else:
    empty_state("🧬", "No ontology built yet for this collection — launch a build above.")
