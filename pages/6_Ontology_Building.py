"""Ontology Building page — OLAF (MCP) + Oxigraph, driven by a tool-calling LLM agent."""
from __future__ import annotations

import os
import uuid

import streamlit as st

from cogmaps.config import ONTOLOGY_TOOLCALL_MODELS, TEMP_GRAPH_OUTPUTS, oxigraph_url
from cogmaps.ontology.graph import build_pyvis_html
from cogmaps.ontology.olaf_config import export_oxigraph_ontology_ttl, sanitize_ontology_id, ttl_path_for
from cogmaps.ontology.runner import OntologyJobRunner
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


# ── Phase 0: Domain Discovery (GLM-5.3-Flash) ───────────────────────────
from cogmaps.ontology.domain_discovery import (
    DomainBlueprint,
    DomainPillar,
    discover_domain,
    load_blueprint,
    save_blueprint,
)


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_lines(text: str) -> list[str]:
    return [line.strip().lstrip("-*• ") for line in text.splitlines() if line.strip()]


blueprint = load_blueprint(qdrant_col)
with st.container(border=True):
    col_d1, col_d2 = st.columns([0.72, 0.28], vertical_alignment="center")
    with col_d1:
        st.markdown("#### 🧬 Phase 0: Domain Blueprint")
        if blueprint:
            st.markdown(
                f"**Inferred Domain:** {blueprint.inferred_domain} &nbsp;·&nbsp; "
                f"`{blueprint.epistemological_nature}`"
            )
        else:
            st.info("No domain blueprint generated yet. One will be synthesized automatically using **GLM-5.3-Flash** upon launch, or you can run discovery or create one below.")
    with col_d2:
        if blueprint:
            if st.button("↻ re-discover", use_container_width=True):
                if not sb.nebius_api_key:
                    st.warning("Set NEBIUS_API_KEY to run domain discovery.")
                else:
                    with st.spinner("Profiling corpus and synthesizing domain pillars with GLM-5.3-Flash..."):
                        try:
                            blueprint = discover_domain(store, qdrant_col, force=True)
                            st.success("Domain discovery complete!")
                            st.rerun()
                        except Exception as err:
                            st.error(f"Domain discovery failed: {err}")
        else:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🔍 discover", use_container_width=True):
                    if not sb.nebius_api_key:
                        st.warning("Set NEBIUS_API_KEY to run domain discovery.")
                    else:
                        with st.spinner("Profiling corpus and synthesizing domain pillars with GLM-5.3-Flash..."):
                            try:
                                blueprint = discover_domain(store, qdrant_col, force=True)
                                st.success("Domain discovery complete!")
                                st.rerun()
                            except Exception as err:
                                st.error(f"Domain discovery failed: {err}")
            with c_btn2:
                if st.button("✏️ create custom", use_container_width=True):
                    blueprint = DomainBlueprint(
                        inferred_domain=f"Custom Domain ({collection_name})",
                        epistemological_nature="Empirical",
                        pillars=[
                            DomainPillar(
                                name="Core Concepts",
                                target_parent_class="Concept",
                                sample_classes=[],
                                typical_relations=[],
                            )
                        ],
                    )
                    save_blueprint(qdrant_col, blueprint)
                    st.rerun()

    if blueprint:
        tab_view, tab_form, tab_json = st.tabs(["👁️ Overview", "✏️ Form Editor", "🧬 Raw JSON"])

        with tab_view:
            for p in blueprint.pillars:
                parent_str = f" *(subClassOf `{p.target_parent_class}`)*" if p.target_parent_class else ""
                st.markdown(f"**• {p.name}**{parent_str}")
                if p.sample_classes:
                    st.caption(f"Concepts: {', '.join(p.sample_classes)}")
                if p.typical_relations:
                    st.caption(f"Relations: {', '.join(p.typical_relations)}")
            if blueprint.individual_types:
                st.markdown(f"**• Individual Types:** {', '.join(blueprint.individual_types)}")
            if blueprint.cross_cutting_themes:
                st.markdown("**• Cross-Cutting Inter-Pillar Themes:**")
                for theme in blueprint.cross_cutting_themes:
                    st.markdown(f"  - {theme}")
            if blueprint.suggested_object_properties:
                st.caption(f"Suggested Object Properties: {', '.join(blueprint.suggested_object_properties)}")

        with tab_form:
            st.caption("Edit the domain properties, taxonomical dimensions, and entity types. Submit to save changes.")
            with st.form(key=f"edit_blueprint_form_{qdrant_col}"):
                new_domain = st.text_input("Inferred Domain", value=blueprint.inferred_domain)
                nature_options = ["Empirical", "Deontic", "Interdisciplinary", "Formal", "Normative"]
                current_nature = blueprint.epistemological_nature if blueprint.epistemological_nature in nature_options else "Interdisciplinary"
                new_nature = st.selectbox(
                    "Epistemological Nature",
                    nature_options,
                    index=nature_options.index(current_nature) if current_nature in nature_options else 2,
                )

                st.markdown("##### Taxonomical Pillars")
                updated_pillars = []
                for idx, pillar in enumerate(blueprint.pillars):
                    with st.expander(f"Pillar {idx + 1}: {pillar.name}", expanded=False):
                        p_name = st.text_input("Pillar Name", value=pillar.name, key=f"p_name_{qdrant_col}_{idx}")
                        p_parent = st.text_input(
                            "Target Parent Class",
                            value=pillar.target_parent_class,
                            key=f"p_parent_{qdrant_col}_{idx}",
                            help="Universal root OWL class to anchor concepts under (e.g. Asset, LegalNorm, CognitiveProcess)",
                        )
                        p_classes_str = st.text_area(
                            "Candidate Concepts (comma-separated)",
                            value=", ".join(pillar.sample_classes),
                            key=f"p_classes_{qdrant_col}_{idx}",
                        )
                        p_relations_str = st.text_input(
                            "Typical Relations (comma-separated)",
                            value=", ".join(pillar.typical_relations),
                            key=f"p_relations_{qdrant_col}_{idx}",
                        )
                        p_remove = st.checkbox("Remove this pillar", key=f"p_remove_{qdrant_col}_{idx}")
                        if not p_remove and p_name.strip():
                            updated_pillars.append(
                                DomainPillar(
                                    name=p_name.strip(),
                                    target_parent_class=p_parent.strip(),
                                    sample_classes=_parse_csv(p_classes_str),
                                    typical_relations=_parse_csv(p_relations_str),
                                )
                            )

                with st.expander("➕ Add New Pillar", expanded=False):
                    new_p_name = st.text_input("New Pillar Name", key=f"new_p_name_{qdrant_col}")
                    new_p_parent = st.text_input("New Target Parent Class", key=f"new_p_parent_{qdrant_col}")
                    new_p_classes = st.text_area("New Candidate Concepts (comma-separated)", key=f"new_p_classes_{qdrant_col}")
                    new_p_relations = st.text_input("New Typical Relations (comma-separated)", key=f"new_p_relations_{qdrant_col}")
                    if new_p_name.strip():
                        updated_pillars.append(
                            DomainPillar(
                                name=new_p_name.strip(),
                                target_parent_class=new_p_parent.strip(),
                                sample_classes=_parse_csv(new_p_classes),
                                typical_relations=_parse_csv(new_p_relations),
                            )
                        )

                st.markdown("##### Entity Categories & Relationships")
                ind_types_str = st.text_area(
                    "Individual Types (comma-separated)",
                    value=", ".join(blueprint.individual_types),
                    help="Categories for owl:NamedIndividual (e.g. Theorist, Agency, Specific Index)",
                )
                obj_props_str = st.text_area(
                    "Suggested Object Properties (comma-separated)",
                    value=", ".join(blueprint.suggested_object_properties),
                    help="Key directed properties connecting classes across pillars",
                )
                cross_cutting_str = st.text_area(
                    "Cross-Cutting Inter-Pillar Themes (one per line)",
                    value="\n".join(blueprint.cross_cutting_themes),
                    help="Thematic bridges linking different pillars to prevent disconnected clusters",
                )

                submit_form = st.form_submit_button("💾 Save Form Changes", type="primary")
                if submit_form:
                    updated_bp = DomainBlueprint(
                        inferred_domain=new_domain.strip(),
                        epistemological_nature=new_nature,
                        pillars=updated_pillars,
                        individual_types=_parse_csv(ind_types_str),
                        suggested_object_properties=_parse_csv(obj_props_str),
                        cross_cutting_themes=_parse_lines(cross_cutting_str),
                    )
                    save_blueprint(qdrant_col, updated_bp)
                    st.success("Domain blueprint updated and saved successfully!")
                    st.rerun()

        with tab_json:
            st.caption("Directly inspect and modify the raw JSON representation of the domain blueprint.")
            with st.form(key=f"edit_blueprint_json_{qdrant_col}"):
                json_content = st.text_area(
                    "Blueprint JSON",
                    value=blueprint.model_dump_json(indent=2),
                    height=420,
                )
                submit_json = st.form_submit_button("💾 Save JSON Changes", type="primary")
                if submit_json:
                    try:
                        new_bp = DomainBlueprint.model_validate_json(json_content)
                        save_blueprint(qdrant_col, new_bp)
                        st.success("Domain blueprint JSON saved successfully!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Invalid DomainBlueprint JSON: {err}")

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

# Live sync with Oxigraph: if Oxigraph has the active named graph with classes,
# ensure the on-disk .ttl matches Oxigraph's source of truth.
ox_ttl = export_oxigraph_ontology_ttl(ontology_id, oxigraph_url())
if ox_ttl and ("owl:Class" in ox_ttl or "a <http://www.w3.org/2002/07/owl#Class>" in ox_ttl):
    with open(ttl_file, "w", encoding="utf-8") as f:
        f.write(ox_ttl)
    ttl = ox_ttl
elif os.path.exists(ttl_file):
    with open(ttl_file, encoding="utf-8") as f:
        ttl = f.read()
else:
    ttl = None

if ttl:
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
