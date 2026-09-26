"""Manage page — list, delete documents per ingestion date."""
from __future__ import annotations

import datetime
import html

import pandas as pd
import streamlit as st

from cogmaps.ontology.cleanup import (
    delete_ontology_for_collection,
    delete_seed,
    list_seed_ids,
    ontology_exists_for,
)
from cogmaps.ontology.olaf_config import sanitize_ontology_id
from cogmaps.ui.auth import (
    check_password,
    is_admin,
    list_visible_collections,
    owns_collection,
)
from cogmaps.ui.components import empty_state, render_sidebar, section_header
from cogmaps.ui.styles import apply_global_styles
from cogmaps.qdrant.store import QdrantStore

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/manage/", "list and delete embedded docs")

if not sb.is_connected:
    empty_state("⚡", "Qdrant offline.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
col_labels = [label for label, _ in _visible]
col_map = {label: qdrant_name for label, qdrant_name in _visible}

if not col_labels:
    empty_state("📂", "No collections found.")
    st.stop()

col_sel, col_date = st.columns([1, 1])
with col_sel:
    collection_name = st.selectbox("Select Collection", col_labels)
with col_date:
    selected_date: datetime.date = st.date_input("Select Ingestion Date", datetime.datetime.today())

qdrant_col = col_map[collection_name]
_is_public = collection_name.startswith("[public] ")
_can_modify = is_admin() if _is_public else owns_collection(qdrant_col)

_ontology_id = sanitize_ontology_id(qdrant_col)
_ontology_exists = ontology_exists_for(qdrant_col)

with st.expander("⚠ danger zone — delete entire collection"):
    if not _can_modify:
        st.warning(
            "You do not have permission to delete this collection. "
            + ("Only admins can delete public collections." if _is_public else "")
        )
    else:
        pt_count, _ = store.collection_stats(qdrant_col)
        if pt_count is not None:
            st.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#a82020">'
                f'This will permanently delete collection <strong>{html.escape(collection_name)}</strong> '
                f'and all its {pt_count:,} vectors.</p>',
                unsafe_allow_html=True,
            )
        # Deleting a collection cascades to its ontology — an ontology is
        # meaningless without the source chunks it was extracted from. The
        # reverse is never true: deleting an ontology (below) never touches
        # the collection it was built from.
        st.warning(
            "⚠ Deleting this collection will also permanently delete the ontology "
            "built for it in Oxigraph (if any), including its concepts collection and "
            "cached Turtle export. This does not go the other way: deleting an ontology "
            "never deletes the collection."
        )
        # Key includes qdrant_col so switching the selected collection doesn't
        # carry over a stale "confirmed" checkbox state to a different one.
        if st.checkbox(
            f"I confirm permanent deletion of '{collection_name}'"
            + (" and its ontology" if _ontology_exists else ""),
            key=f"confirm_del_col_{qdrant_col}",
        ):
            if st.button("delete collection now", type="primary"):
                if _ontology_exists:
                    try:
                        for msg in delete_ontology_for_collection(qdrant_col, sb.qdrant_host, sb.qdrant_port):
                            st.caption(f"✓ {msg}")
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"Could not fully delete the associated ontology: {e}")
                try:
                    store.delete_collection(qdrant_col)
                    st.success(f"'{collection_name}' deleted.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

with st.expander("⚠ danger zone — delete ontology only"):
    if not _can_modify:
        st.warning(
            "You do not have permission to delete this ontology. "
            + ("Only admins can delete ontologies built on public collections." if _is_public else "")
        )
    elif not _ontology_exists:
        st.caption(f"No ontology has been built for '{collection_name}' yet.")
    else:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#a82020">'
            f'This will permanently delete the ontology for <strong>{html.escape(collection_name)}</strong> '
            f'(named graph <code>urn:olaf:{_ontology_id}</code>) from Oxigraph, its concepts collection, '
            f'and the cached Turtle export. <strong>The Qdrant collection itself is never affected.</strong></p>',
            unsafe_allow_html=True,
        )
        if st.checkbox(
            f"I confirm permanent deletion of the ontology for '{collection_name}'",
            key=f"confirm_del_onto_{qdrant_col}",
        ):
            if st.button("delete ontology now", type="primary", key=f"del_onto_btn_{qdrant_col}"):
                try:
                    for msg in delete_ontology_for_collection(qdrant_col, sb.qdrant_host, sb.qdrant_port):
                        st.caption(f"✓ {msg}")
                    st.success("Ontology deleted.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

with st.expander("🌱 danger zone — delete seed ontologies (global)"):
    st.caption(
        "Seeds are reference ontologies shared across every collection's ontology "
        "in Oxigraph — not scoped to the collection selected above. Deleting one "
        "affects every ontology that reuses it."
    )
    if not is_admin():
        st.warning("Only admins can delete seed ontologies.")
    else:
        seed_ids = list_seed_ids()
        if not seed_ids:
            st.caption("No seed ontologies loaded.")
        else:
            seed_to_delete = st.selectbox("seed", seed_ids, key="seed_to_delete")
            st.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#a82020">'
                f'This will permanently delete seed <strong>{html.escape(seed_to_delete)}</strong> '
                f'(named graph <code>urn:olaf:seed:{html.escape(seed_to_delete)}</code>) from Oxigraph.</p>',
                unsafe_allow_html=True,
            )
            if st.checkbox(
                f"I confirm permanent deletion of seed '{seed_to_delete}'",
                key=f"confirm_del_seed_{seed_to_delete}",
            ):
                if st.button("delete seed now", type="primary", key=f"del_seed_btn_{seed_to_delete}"):
                    try:
                        for msg in delete_seed(seed_to_delete, sb.qdrant_host, sb.qdrant_port):
                            st.caption(f"✓ {msg}")
                        st.success(f"Seed '{seed_to_delete}' deleted.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error: {e}")

st.info(f"Viewing documents in **{collection_name}** ingested on **{selected_date.isoformat()}**")
with st.spinner("fetching documents..."):
    docs = store.documents_for_date(qdrant_col, selected_date)

if not docs:
    empty_state("💨", "No documents found for this date.")
else:
    st.dataframe(
        pd.DataFrame([{"Filename": k, "Chunks": v} for k, v in docs.items()]),
        use_container_width=True,
    )

    if not _can_modify:
        st.info("Read-only view — you do not have permission to delete documents from this collection.")
    else:
        st.markdown("---")
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
            'color:#a82020;text-transform:uppercase;letter-spacing:0.1em">⚠️ danger zone</p>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Delete ALL for {selected_date.isoformat()}", type="primary"):
                try:
                    store.delete_for_date(qdrant_col, selected_date)
                    st.success(f"Deleted all points for {selected_date.isoformat()} in {collection_name}")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

        with c2:
            doc_to_delete = st.selectbox("Select specific document to delete", [""] + sorted(docs.keys()))
            if doc_to_delete and st.button(f"Delete '{doc_to_delete}'"):
                try:
                    store.delete_document_on_date(qdrant_col, doc_to_delete, selected_date)
                    st.success(f"Deleted '{doc_to_delete}'")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")
