"""Deletes an ontology (or a global seed) and everything OLAF/cog-maps derived from it.

Used by the Manage page (``pages/7_Manage.py``):
  - deleting an ontology never touches the Qdrant collection it was built from,
  - but deleting a Qdrant collection *does* cascade to delete its ontology —
    an ontology is meaningless without the source chunks it was extracted
    from, so leaving it behind would just be orphaned data.
  - seeds are global reference ontologies (shared across every ontology in
    the store, see OLAF's README) — deleting one is independent of any
    collection.

Runs OLAF's own CLI drop commands (``olaf drop <ontology_id>`` /
``olaf drop-seed <seed_id>``) through :mod:`cogmaps.ontology.olaf_cli`. Both
remove the Oxigraph graph and its semantic index (the ``olaf_index_{id}`` /
``olaf_index_seed_{id}`` Qdrant collection). The rest is cleaned up here
directly: the ``olaf_concepts_{id}`` collection an ontology built by an older
OLAF may still have, the locally cached Turtle export, the reasoning backups
and the collection's cached domain blueprint (see
:mod:`cogmaps.ontology.domain_discovery`).
"""
from __future__ import annotations

import os
import shutil

from cogmaps.config import oxigraph_url as cfg_oxigraph_url
from cogmaps.ontology.domain_discovery import delete_blueprint
from cogmaps.ontology.olaf_cli import run_olaf, scratch_config_dir
from cogmaps.ontology.olaf_config import (
    LEGACY_CONCEPTS_PREFIX,
    backup_dir_for,
    sanitize_ontology_id,
    ttl_path_for,
)
from cogmaps.ontology.oxigraph_client import is_reachable, list_graphs
from cogmaps.ontology.seeds import SEED_GRAPH_PREFIX
from cogmaps.qdrant.store import QdrantStore


def ontology_exists_for(collection: str) -> bool:
    """Whether an ontology graph currently exists in Oxigraph for this collection.

    Returns False (rather than raising) when Oxigraph is unreachable —
    ontology cleanup is a secondary concern on this page; a Qdrant-only
    Manage page must keep working when Oxigraph happens to be down.
    """
    url = cfg_oxigraph_url()
    if not is_reachable(url):
        return False
    graph_uri = f"urn:olaf:{sanitize_ontology_id(collection)}"
    try:
        return graph_uri in list_graphs(url)
    except Exception:  # noqa: BLE001
        return False


def delete_ontology_for_collection(collection: str, qdrant_host: str, qdrant_port: int) -> list[str]:
    """Drop the Oxigraph ontology graph + its semantic index (and legacy concepts collection)
    + cached .ttl export + reasoning backups + domain blueprint.

    Returns human-readable status lines for each step actually performed.
    Raises ``RuntimeError`` if the ``olaf drop`` subprocess itself fails.
    """
    ontology_id = sanitize_ontology_id(collection)
    messages: list[str] = []

    with scratch_config_dir(
        qdrant_url=f"http://{qdrant_host}:{qdrant_port}", qdrant_collection=collection,
        ontology_id=ontology_id, ontology_name=collection,
    ) as config_dir:
        run_olaf(["drop", ontology_id], config_dir)
        messages.append(f"Ontology graph 'urn:olaf:{ontology_id}' and its semantic index dropped.")

    store = QdrantStore(qdrant_host, qdrant_port)
    legacy_collection = f"{LEGACY_CONCEPTS_PREFIX}_{ontology_id}"
    if legacy_collection in store.list_collections():
        store.delete_collection(legacy_collection)
        messages.append(f"Legacy concepts collection '{legacy_collection}' deleted.")

    ttl_file = ttl_path_for(collection)
    if os.path.exists(ttl_file):
        os.remove(ttl_file)
        messages.append("Cached Turtle export removed.")

    backup_dir = backup_dir_for(collection)
    if backup_dir.is_dir():
        shutil.rmtree(backup_dir)
        messages.append("Reasoning backups removed.")

    if delete_blueprint(collection):
        messages.append("Domain blueprint removed.")

    return messages


def list_seed_ids() -> list[str]:
    """Global seed ontology ids currently loaded in Oxigraph (``urn:olaf:seed:{id}``).

    Seeds are shared across every ontology in the store — not scoped to any
    one Qdrant collection. Returns an empty list (rather than raising) when
    Oxigraph is unreachable, same reasoning as :func:`ontology_exists_for`.
    """
    url = cfg_oxigraph_url()
    if not is_reachable(url):
        return []
    try:
        graphs = list_graphs(url)
    except Exception:  # noqa: BLE001
        return []
    return sorted(g[len(SEED_GRAPH_PREFIX):] for g in graphs if g.startswith(SEED_GRAPH_PREFIX))


def delete_seed(seed_id: str, qdrant_host: str, qdrant_port: int) -> list[str]:
    """Drop a global seed ontology graph and its semantic index via ``olaf drop-seed <seed_id>``."""
    with scratch_config_dir(qdrant_url=f"http://{qdrant_host}:{qdrant_port}") as config_dir:
        run_olaf(["drop-seed", seed_id], config_dir)
    return [f"Seed 'urn:olaf:seed:{seed_id}' and its semantic index dropped."]
