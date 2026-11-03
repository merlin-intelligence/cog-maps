"""Ontology Building page — OLAF (MCP) + Oxigraph, driven by a tool-calling LLM agent."""
from __future__ import annotations

import os
import uuid

import streamlit as st

from cogmaps.config import ONTOLOGY_TOOLCALL_MODELS, TEMP_GRAPH_OUTPUTS, domain_discovery_model, oxigraph_url
from cogmaps.core.llm_impacts import LLMCallImpact, record_llm_impacts
from cogmaps.ontology.domain_discovery import (
    discover_domain,
    load_blueprint,
    load_manual_guidelines,
    save_blueprint,
    save_manual_guidelines,
)
from cogmaps.ontology.cleanup import ontology_ids_for
from cogmaps.ontology.graph import build_pyvis_html
from cogmaps.ontology.olaf_config import (
    export_oxigraph_ontology_ttl,
    has_ontology_content,
    list_backups,
    ontology_id_for,
    ontology_name_of,
    restore_oxigraph_graph,
    sanitize_ontology_id,
    ttl_path_for,
)
from cogmaps.ontology.prompts import format_domain_guidelines
from cogmaps.ontology.runner import OntologyJobRunner, build_domain_mode, build_seed_ids
from cogmaps.ontology.seeds import list_seeds
from cogmaps.ontology.store import OntologyJobStore
from cogmaps.qdrant.store import QdrantStore
from cogmaps.ui.auth import check_password, is_admin, list_visible_collections
from cogmaps.ui.components import (
    empty_state,
    render_llm_impacts_footer,
    render_sidebar,
    section_header,
    session_llm_calls,
)
from cogmaps.ui.styles import apply_global_styles

# ── Module-level singletons (one per server process, shared across sessions) ──

@st.cache_resource
def _ontology_job_store() -> OntologyJobStore:
    os.makedirs("user_data", exist_ok=True)
    return OntologyJobStore(os.path.join("user_data", "jobs.db"))


@st.cache_resource
def _ontology_job_runner(qdrant_host: str, qdrant_port: int) -> OntologyJobRunner:
    return OntologyJobRunner(_ontology_job_store(), qdrant_host, qdrant_port)


@st.cache_data(ttl=15, show_spinner=False)
def _oxigraph_ttl(ontology_id: str, url: str) -> str | None:
    """Current Turtle of the ontology's named graph in Oxigraph, if it has content.

    Cached briefly so ordinary page interactions don't each hit Oxigraph; a
    short timeout keeps the page responsive when Oxigraph is down.
    """
    ttl = export_oxigraph_ontology_ttl(ontology_id, url, timeout=3)
    return ttl if has_ontology_content(ttl) else None


@st.cache_data(ttl=15, show_spinner=False)
def _oxigraph_ontology_ids(collection: str) -> list[str]:
    """The collection's ontologies in Oxigraph, cached like :func:`_oxigraph_ttl`."""
    return ontology_ids_for(collection)


def _show_results() -> None:
    """Refetch the ontology and rerun the whole page.

    Called from inside the job-status fragment, where a plain rerun would only
    redraw the fragment — not the results section below it.
    """
    _oxigraph_ttl.clear()
    _oxigraph_ontology_ids.clear()
    st.rerun(scope="app")


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
# A collection can have several ontologies: its default one, and others named here.
_default_ontology_id = sanitize_ontology_id(qdrant_col)
_NEW_ONTOLOGY = "+ new ontology"
_ontology_key = f"ontology_{qdrant_col}"
_ontology_ids = list(dict.fromkeys([
    _default_ontology_id,
    *_oxigraph_ontology_ids(qdrant_col),
    # also those of builds not exported yet (running, or failed before their first concept)
    *_ontology_job_store().ontology_ids_for(qdrant_col),
]))
# Set by a build launch (once the selectbox is drawn, its value can't change in that run).
if _launched := st.session_state.pop(f"{_ontology_key}_next", None):
    st.session_state[_ontology_key] = _launched
# A radio, not a selectbox: a selectbox's search box reads as a place to type a name.
_ontology_choice = st.radio(
    "ontology",
    [*_ontology_ids, _NEW_ONTOLOGY],
    format_func=lambda oid: oid if oid == _NEW_ONTOLOGY else (ontology_name_of(qdrant_col, oid) or "default"),
    horizontal=True,
    key=_ontology_key,
    help="Each ontology of the collection is built separately — its own graph, and its own progress "
         "through the chunks. Create a new one to build another version from the same documents.",
)
if _ontology_choice == _NEW_ONTOLOGY:
    _new_name = st.text_input(
        "new ontology name", placeholder="e.g. v2, legal…", key=f"new_ontology_name_{qdrant_col}",
        help="Letters and digits; other characters become '_'.",
    )
    ontology_id = ontology_id_for(qdrant_col, _new_name)
    if ontology_id == _default_ontology_id:
        st.info("Name the new ontology to build it.")
        st.stop()
    if ontology_id in _ontology_ids:
        st.warning(f"The collection already has an ontology named “{ontology_name_of(qdrant_col, ontology_id)}”: "
                   "a build will extend it.")
else:
    ontology_id = _ontology_choice
st.caption(f"ontology id: `{ontology_id}` — repeated builds on this ontology extend it")

filenames = sorted(store.existing_filenames(qdrant_col))

# ── Job helpers ───────────────────────────────────────────────────────

# Builds and reasoning runs each have their own active job, shown in their own
# section of the page: "build" above the results, "reasoning" in the Reasoning section.
_JOB_KINDS = {"build": "ontology build", "reasoning": "reasoning"}


def _active_job_key(job_type: str) -> str:
    return f"ontology_active_{job_type}_job_id"


def _activate_job(job_id: str, job_type: str) -> None:
    st.session_state[_active_job_key(job_type)] = job_id


def _job_log(job_id: str) -> str:
    """The job's log so far. Kept per job in the session, so each refresh only fetches
    the new lines."""
    key = f"ontology_log_{job_id}"
    cursor, buf = st.session_state.get(key, (0, ""))
    new_entries = _ontology_job_store().get_logs(job_id, after_rowid=cursor)
    if new_entries:
        rowids, messages = zip(*new_entries, strict=True)
        cursor, buf = rowids[-1], buf + "\n".join(messages) + "\n"
        st.session_state[key] = (cursor, buf)
    return buf


_PROBLEM_COLUMNS = {
    "unsatisfiable": "unsatisfiable classes",
    "cycles": "classes in subclass cycles",
    "domain": "domain violations",
    "range": "range violations",
    "untyped": "untyped individuals",
    "kind": "property kind mismatches",
    "unknown_terms": "unknown vocabulary terms",
    "orphans": "orphans",
}


def _render_reasoning_result(result: dict) -> None:
    """What a reasoning run did: merges, disjointness, inferences, and the problem counts of
    every check (before the fixes, after each round)."""
    if isinstance(result, list):  # runs recorded before dedup/enrich/infer: the checks only
        result = {"checks": result}
    history = result.get("checks") or []
    inferences = result.get("inferences")
    cols = st.columns(4)
    cols[0].metric("changes applied", result.get("changes", 0))
    cols[1].metric("duplicate candidates", result.get("dedup_pairs", 0))
    cols[2].metric("disjoint pairs declared", result.get("disjoint_added", 0))
    cols[3].metric(
        "inferences materialized",
        "—" if not inferences or "materialized" not in inferences else inferences["materialized"],
        help="Triples the reasoner entails but the ontology doesn't assert, written into it marked as "
             "inferred (no source chunk). Shown as dotted edges in the graph view.",
    )
    if not history:
        return
    rows = [
        {
            "check": h["label"],
            "consistent": "?" if h["consistent"] is None else ("yes" if h["consistent"] else "no"),
            **{label: h.get(key, 0) for key, label in _PROBLEM_COLUMNS.items()},
        }
        for h in history
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if any(h.get("reasoner_error") for h in history):
        st.warning(
            "The OWL reasoner (Pellet) could not run — only the SPARQL integrity checks did. "
            "It needs a Java 11+ runtime on the server; see the reasoning log for the error."
        )


def _job_to_show(job_type: str) -> str | None:
    """The job a section's log window shows: the one launched from this session, if it is
    about the selected collection, otherwise the collection's latest job of that type — so
    a running job is picked up again, and a finished one's log is still there, after a
    reload or a tab closure."""
    job_id = st.session_state.get(_active_job_key(job_type))
    if job_id:
        job = _ontology_job_store().get_job(job_id)
        if job and job["collection"] == qdrant_col and job["ontology_id"] == ontology_id:
            return job_id
    latest = _ontology_job_store().latest_job_for(qdrant_col, job_type, ontology_id)
    return latest["id"] if latest else None


@st.fragment(run_every=2.0)
def _render_job_status(job_type: str) -> None:
    job_id = _job_to_show(job_type)
    if not job_id:
        return
    job = _ontology_job_store().get_job(job_id)
    if not job:
        return

    status = job["status"]
    is_reasoning = job_type == "reasoning"
    icons = {"pending": "⏳", "running": "⚙️", "done": "✅", "failed": "❌"}
    st.markdown(
        f"**{icons.get(status, '·')} {_JOB_KINDS[job_type]}** — `{status}` "
        f"<span style='opacity:0.6'>· started {job['created_at'][:16].replace('T', ' ')} · {job['llm_model']}</span>",
        unsafe_allow_html=True,
    )

    cur, tot = job["progress_current"], job["progress_total"]
    if status in ("pending", "running"):
        st.progress(min(cur / tot, 1.0) if tot > 0 else 0)
        if tot > 0:
            st.caption(f"{'step' if is_reasoning else 'iteration'} {cur} / {tot}")

    log = _job_log(job_id)
    if log:
        with st.expander(f"{'reasoning' if is_reasoning else 'build'} log", expanded=(status == "running")):
            # Fixed height: the log scrolls instead of pushing the page down.
            with st.container(height=420):
                st.code(log.strip(), language=None)
            st.download_button(
                "download log", log.encode("utf-8"),
                file_name=f"{job_type}-{job['created_at'][:19].replace(':', '')}.log", mime="text/plain",
                key=f"download_log_{job_id}",
            )

    if status == "done":
        if is_reasoning:
            st.success("Reasoning complete!")
            _render_reasoning_result(job["result"] or {})
        else:
            st.success("Ontology build complete!")
    elif status == "failed":
        st.error(f"{'Reasoning' if is_reasoning else 'Ontology build'} failed — see the log above.")

    # Once per job, refresh the results section when it ends: the ontology changed. Only
    # for a job watched while it ran — not for an old run shown again after a reload.
    shown_key = "ontology_results_shown_for"
    if status in ("pending", "running"):
        st.session_state.setdefault("ontology_watched_jobs", set()).add(job_id)
    elif job_id in st.session_state.get("ontology_watched_jobs", set()) and \
            st.session_state.get(shown_key) != job_id:
        st.session_state[shown_key] = job_id
        _show_results()


# ── Domain discovery ────────────────────────────────────────────────────

_discovery_model = domain_discovery_model()


def _blueprint_version_key(collection: str) -> str:
    return f"ontology_blueprint_version_{collection}"


def _blueprint_changed(message: str | None = None) -> None:
    """Call after the saved blueprint changes, right before st.rerun().

    Bumps the version embedded in the editor widget's key so it is rebuilt
    from the saved blueprint — otherwise Streamlit keeps the text previously
    typed instead of showing, say, a fresh discovery.
    Also queues ``message`` to display after the rerun.
    """
    key = _blueprint_version_key(qdrant_col)
    st.session_state[key] = st.session_state.get(key, 0) + 1
    if message:
        st.session_state["ontology_blueprint_flash"] = message


def _run_discovery() -> None:
    if not sb.scaleway_api_key:
        st.warning("Set SCALEWAY_API_KEY to run domain discovery.")
        return
    with st.spinner(f"Profiling corpus and synthesizing domain pillars with {_discovery_model}..."):
        try:
            with record_llm_impacts(session_llm_calls("ontology_building").append):
                discover_domain(store, qdrant_col, model=_discovery_model, force=True)
        except Exception as err:  # noqa: BLE001
            st.error(f"Domain discovery failed: {err}")
            return
    _blueprint_changed("Domain discovery complete!")
    st.rerun()


# Starting point of guidelines written by hand: the shape of the generated ones.
_GUIDELINES_TEMPLATE = """When analyzing <your domain> text, systematically identify and extract:
1. **<First dimension> (`owl:Class` subClassOf <ParentClass>):**
   Candidate concepts: <Concept A>, <Concept B>, <Concept C>.
   Expected relations: <relationA>, <relationB>.
2. **<Second dimension> (`owl:Class` subClassOf <ParentClass>):**
   Candidate concepts: ...
3. **Key Institutions & Specific Named Individuals (`owl:NamedIndividual`):**
   Concrete named entities with unique identity, e.g. <entity categories>.
   Use `individual_create` with the URI of the `owl:Class` this entity is an instance of.

### Cross-Pillar Relationships & Interconnections:
- <theme linking two dimensions>
"""


_DOMAIN_MODES = {
    "none": "none — default prompt",
    "llm": "LLM discovery",
    "manual": "written by hand",
}

_flash_msg = st.session_state.pop("ontology_blueprint_flash", None)
# Prefix for every editor widget key; changes whenever the saved blueprint does.
_bp_key = f"{qdrant_col}_v{st.session_state.get(_blueprint_version_key(qdrant_col), 0)}"
with st.container(border=True):
    if _flash_msg:
        st.success(_flash_msg)
    st.markdown("#### 🧬 Domain guidance")
    domain_mode = st.radio(
        "domain guidance",
        list(_DOMAIN_MODES),
        index=list(_DOMAIN_MODES).index(build_domain_mode(_ontology_job_store(), qdrant_col, ontology_id)),
        format_func=_DOMAIN_MODES.get,
        horizontal=True,
        label_visibility="collapsed",
        key=f"domain_mode_{ontology_id}",
        help="Optional text added to the agent's prompt to tailor it to this corpus. "
             "Defaults to the choice of this ontology's last build.",
    )

    if domain_mode == "none":
        st.caption("The agent uses the default, domain-neutral prompt.")

    elif domain_mode == "llm":
        blueprint = load_blueprint(qdrant_col)
        col_d1, col_d2 = st.columns([0.72, 0.28], vertical_alignment="center")
        with col_d1:
            if blueprint:
                st.markdown(
                    f"**Inferred Domain:** {blueprint.inferred_domain} &nbsp;·&nbsp; "
                    f"`{blueprint.epistemological_nature}`"
                )
                if blueprint.generated_at:
                    st.caption(
                        f"discovered {blueprint.generated_at[:16].replace('T', ' ')} "
                        f"from {blueprint.source_document_count} document(s)"
                        + (" · ✍️ edited by hand" if blueprint.guidelines else "")
                    )
                if blueprint.source_document_count is not None and blueprint.source_document_count != len(filenames):
                    st.warning(
                        f"The collection now has {len(filenames)} document(s) — "
                        "this blueprint may be out of date. Consider re-discovering."
                    )
            else:
                st.caption(
                    f"Not discovered yet: run it now to review and edit the result, or it runs "
                    f"at launch (**{_discovery_model}**)."
                )
        with col_d2:
            if blueprint:
                # Replaces the text, hand edits included: confirm first.
                with st.popover("↻ re-discover", use_container_width=True):
                    st.caption("Replaces the current text — and your edits — with a new LLM discovery.")
                    if st.button("confirm re-discover", key=f"bp_rediscover_{_bp_key}"):
                        _run_discovery()
            elif st.button("🔍 discover", use_container_width=True):
                _run_discovery()

        if blueprint:
            with st.form(key=f"edit_blueprint_text_{_bp_key}"):
                guidelines_text = st.text_area(
                    "Discovered guidelines — edit them freely (markdown)",
                    value=format_domain_guidelines(blueprint),
                    height=420,
                    key=f"bp_text_{_bp_key}",
                )
                if st.form_submit_button("💾 Save", type="primary"):
                    blueprint.guidelines = guidelines_text.strip() or None
                    save_blueprint(qdrant_col, blueprint)
                    _blueprint_changed("Domain guidelines saved!")
                    st.rerun()

    else:
        with st.form(key=f"edit_manual_guidelines_{_bp_key}"):
            manual_text = st.text_area(
                "Your guidelines (markdown)",
                value=load_manual_guidelines(qdrant_col) or _GUIDELINES_TEMPLATE,
                height=420,
                key=f"manual_text_{_bp_key}",
            )
            if st.form_submit_button("💾 Save", type="primary"):
                save_manual_guidelines(qdrant_col, manual_text.strip())
                _blueprint_changed("Domain guidelines saved!")
                st.rerun()
        if not load_manual_guidelines(qdrant_col):
            st.caption("Not saved yet — the build only uses saved text.")

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
    try:
        available_seeds = [s.seed_id for s in list_seeds()]
    except Exception as e:  # noqa: BLE001
        available_seeds = []
        st.warning(f"Could not list the seed ontologies: {e}")
    if available_seeds:
        last_seeds = build_seed_ids(_ontology_job_store(), qdrant_col, ontology_id)
        selected_seeds = st.multiselect(
            "seed ontologies to reuse",
            available_seeds,
            default=[s for s in last_seeds if s in available_seeds],
            help="Reference ontologies the agent searches and reuses terms from (classes, properties, "
                 "individuals), under their own URIs — new classes become subclasses of seed classes where "
                 "that is true. Defaults to the seeds of this ontology's last build. "
                 "Load seeds from /explore ontology/.",
        )
    else:
        selected_seeds = []
        st.caption("No seed ontology loaded — load reference ontologies from /explore ontology/ to reuse them here.")

    if not sb.scaleway_api_key:
        st.warning("Set SCALEWAY_API_KEY to run the ontology-building agent (Scaleway tool-calling only).")
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
                llm_provider="scaleway",
                llm_model=model,
                use_domain_discovery=domain_mode != "none",
                options={"seed_ids": selected_seeds, "domain_mode": domain_mode},
            )
            _ontology_job_runner(sb.qdrant_host, sb.qdrant_port).submit(job_id)
            _activate_job(job_id, "build")
            # A new ontology is now one of the collection's: select it.
            st.session_state[f"{_ontology_key}_next"] = ontology_id
            st.rerun()

_render_job_status("build")

st.markdown("---")

# ── Results: raw Turtle + navigable graph ───────────────────────────────

ttl_file = ttl_path_for(qdrant_col, ontology_id)

# Oxigraph is the source of truth; the on-disk .ttl (written by the build
# runner) is only a fallback when Oxigraph is unreachable.
ttl = _oxigraph_ttl(ontology_id, oxigraph_url())
if ttl is None and os.path.exists(ttl_file):
    with open(ttl_file, encoding="utf-8") as f:
        ttl = f.read()

def _render_reasoning() -> None:
    """Launch the reasoning agent on the built ontology, and restore a pre-reasoning backup."""
    with st.container(border=True):
        st.markdown("#### 🧠 Reasoning")
        st.caption(
            "Curates the ontology with the Pellet OWL reasoner and an agent working from the source chunks: "
            "merges duplicate classes, repairs logical problems (inconsistency, unsatisfiable classes, "
            "subclass cycles, domain/range violations, untyped individuals, unknown terms, orphans), "
            "declares disjoint sibling classes, then reviews what the ontology entails and writes it in, "
            "marked as inferred. The ontology is backed up before the first change."
        )
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            reasoning_model = st.selectbox("model (tool-calling)", ONTOLOGY_TOOLCALL_MODELS, key="reasoning_model")
        with col_r2:
            max_rounds = st.number_input(
                "max rounds", min_value=1, max_value=10, value=3,
                help="Check → fix → check again, up to this many times. The run stops early when "
                     "nothing is left or a round changed nothing.",
            )
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            dedup = st.checkbox(
                "merge duplicates", value=True,
                help="First, pairs of classes with near-identical labels/definitions are reviewed: merge or "
                     "keep. Needs OLAF's concept embeddings; skipped otherwise.",
            )
            fix_orphans = st.checkbox(
                "also connect orphans", value=True,
                help="Classes and individuals with no relation to the rest of the ontology — connected only "
                     "when the source text says explicitly what they are.",
            )
            enrich = st.checkbox(
                "declare disjoint classes", value=True,
                help="Once the ontology is consistent, sibling classes that can never share an instance are "
                     "declared disjoint — what lets the reasoner detect contradictions. Checked again afterwards.",
            )
        with col_o2:
            infer = st.checkbox(
                "materialize inferences", value=True,
                help="Once the ontology is consistent, what it entails but doesn't assert (indirect "
                     "superclasses, inherited types, consequences of domains/ranges…) is written in, "
                     "marked as inferred. Needs the OWL reasoner (Java).",
            )
            review = st.checkbox(
                "review inferences first", value=True, disabled=not infer,
                help="The agent reads each new inference with why it holds: an absurd one reveals a wrong "
                     "axiom, which it fixes before anything is written. Unticked, inferences are written as is.",
            )

        reasoning_seeds = build_seed_ids(_ontology_job_store(), qdrant_col, ontology_id)
        st.caption(
            f"seeds (from the last build): {', '.join(reasoning_seeds)}" if reasoning_seeds
            else "seeds: none (the last build used no seed ontology)"
        )

        latest = _ontology_job_store().latest_job_for(qdrant_col, ontology_id=ontology_id)
        busy = latest is not None and latest["status"] in ("running", "pending")
        if not sb.scaleway_api_key:
            st.warning("Set SCALEWAY_API_KEY to run the reasoning agent (Scaleway tool-calling only).")
        elif st.button("▶ launch reasoning", type="primary", disabled=busy,
                       help="A job is already running for this ontology." if busy else None):
            job_id = str(uuid.uuid4())
            _ontology_job_store().create_job(
                job_id=job_id,
                collection=qdrant_col,
                ontology_id=ontology_id,
                doc_filenames=[],
                llm_provider="scaleway",
                llm_model=reasoning_model,
                job_type="reasoning",
                options={
                    "max_rounds": int(max_rounds), "fix_orphans": fix_orphans, "dedup": dedup,
                    "enrich_disjointness": enrich, "infer": infer, "review_inferences": infer and review,
                },
            )
            _ontology_job_runner(sb.qdrant_host, sb.qdrant_port).submit(job_id)
            _activate_job(job_id, "reasoning")
            st.rerun()

        _render_job_status("reasoning")

        backups = list_backups(qdrant_col, ontology_id)
        if backups:
            with st.expander(f"backups before reasoning ({len(backups)})"):
                labels = {
                    f"{b.stem[:4]}-{b.stem[4:6]}-{b.stem[6:8]} {b.stem[9:11]}:{b.stem[11:13]}:{b.stem[13:15]}": b
                    for b in backups
                }
                choice = st.selectbox("backup", list(labels))
                backup_ttl = labels[choice].read_text(encoding="utf-8")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.download_button(
                        "download .ttl", backup_ttl.encode("utf-8"),
                        file_name=f"{ontology_id}-{labels[choice].stem}.ttl", mime="text/turtle",
                        use_container_width=True,
                    )
                with col_b2:
                    if st.button("↺ restore this backup", disabled=busy, use_container_width=True):
                        try:
                            restore_oxigraph_graph(ontology_id, backup_ttl, oxigraph_url())
                        except Exception as err:  # noqa: BLE001
                            st.error(f"Restore failed: {err}")
                        else:
                            # Keep the on-disk fallback copy in step with Oxigraph.
                            os.makedirs(os.path.dirname(ttl_file), exist_ok=True)
                            with open(ttl_file, "w", encoding="utf-8") as f:
                                f.write(backup_ttl)
                            _oxigraph_ttl.clear()
                            st.session_state["ontology_restore_flash"] = f"Ontology restored to the backup of {choice}."
                            st.rerun()
                st.caption("Restoring replaces the ontology graph in Oxigraph with the backup.")
        if flash := st.session_state.pop("ontology_restore_flash", None):
            st.success(flash)


if ttl:
    _render_reasoning()

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
        show_inferred = st.toggle(
            "show inferences", value=True,
            help="Relations materialized by the reasoning step, drawn dotted.",
        )
        try:
            build_pyvis_html(ttl, graph_html_path, show_inferred=show_inferred)
            with open(graph_html_path, encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=850, scrolling=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not render the graph view: {e}")
else:
    empty_state("🧬", "This ontology isn't built yet — launch a build above.")


# ── LLM impacts footer ────────────────────────────────────────────────

@st.fragment(run_every=5.0)
def _render_llm_impacts() -> None:
    """This session's discovery requests plus those of the jobs shown on the page — the
    build and the reasoning run (refreshed while they run)."""
    calls = list(session_llm_calls("ontology_building"))
    for job_type in _JOB_KINDS:
        if job_id := _job_to_show(job_type):
            calls += [LLMCallImpact.from_dict(d) for d in _ontology_job_store().get_llm_calls(job_id)]
    calls.sort(key=lambda c: c.timestamp)
    render_llm_impacts_footer(calls, key="ontology_building")


_render_llm_impacts()
