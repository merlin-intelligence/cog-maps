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
``olaf drop-seed <seed_id>`` — see
https://github.com/merlin-intelligence/olaf, "CLI commands") as short-lived
subprocesses, the same way :mod:`cogmaps.ontology.runner` spawns ``olaf`` for
a build job. ``olaf drop`` only touches Oxigraph, not the
``olaf_concepts_{id}`` Qdrant collection it created for semantic search —
that one, and the locally cached Turtle export, are cleaned up here
directly, along with the collection's cached domain blueprint (see
:mod:`cogmaps.ontology.domain_discovery`). ``olaf drop-seed`` has no Qdrant-side data to clean up (no
concepts collection is ever created for a seed).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from cogmaps.config import oxigraph_url as cfg_oxigraph_url
from cogmaps.config import qdrant_api_key as cfg_qdrant_api_key
from cogmaps.ontology.domain_discovery import delete_blueprint
from cogmaps.ontology.olaf_config import (
    CONCEPTS_COLLECTION_PREFIX,
    backup_dir_for,
    build_config_toml,
    sanitize_ontology_id,
    ttl_path_for,
)
from cogmaps.ontology.oxigraph_client import is_reachable, list_graphs
from cogmaps.qdrant.store import QdrantStore

_DROP_TIMEOUT = 60
_SEED_GRAPH_PREFIX = "urn:olaf:seed:"


@contextmanager
def _scratch_config_dir(
    *, qdrant_host: str, qdrant_port: int, qdrant_collection: str, ontology_id: str, ontology_name: str,
) -> Iterator[str]:
    """A temp dir holding a config.toml for a one-off ``olaf`` CLI invocation."""
    with tempfile.TemporaryDirectory(prefix=f"cogmaps_olaf_{ontology_id}_") as config_dir:
        config_toml = build_config_toml(
            qdrant_url=f"http://{qdrant_host}:{qdrant_port}",
            qdrant_collection=qdrant_collection,
            qdrant_api_key=cfg_qdrant_api_key(),
            oxigraph_url=cfg_oxigraph_url(),
            ontology_id=ontology_id,
            ontology_name=ontology_name,
        )
        with open(os.path.join(config_dir, "config.toml"), "w", encoding="utf-8") as f:
            f.write(config_toml)
        yield config_dir


def _run_olaf(args: list[str], config_dir: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["olaf", *args], cwd=config_dir, capture_output=True, text=True, timeout=_DROP_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"'olaf {' '.join(args)}' failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


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
    """Drop the Oxigraph ontology graph + its concepts Qdrant collection + cached .ttl export
    + reasoning backups + domain blueprint.

    Returns human-readable status lines for each step actually performed.
    Raises ``RuntimeError`` if the ``olaf drop`` subprocess itself fails.
    """
    ontology_id = sanitize_ontology_id(collection)
    messages: list[str] = []

    with _scratch_config_dir(
        qdrant_host=qdrant_host, qdrant_port=qdrant_port, qdrant_collection=collection,
        ontology_id=ontology_id, ontology_name=collection,
    ) as config_dir:
        _run_olaf(["drop", ontology_id], config_dir)
        messages.append(f"Ontology graph 'urn:olaf:{ontology_id}' dropped.")

    store = QdrantStore(qdrant_host, qdrant_port)
    concepts_collection = f"{CONCEPTS_COLLECTION_PREFIX}_{ontology_id}"
    if concepts_collection in store.list_collections():
        store.delete_collection(concepts_collection)
        messages.append(f"Concepts collection '{concepts_collection}' deleted.")

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
    return sorted(g[len(_SEED_GRAPH_PREFIX):] for g in graphs if g.startswith(_SEED_GRAPH_PREFIX))


def delete_seed(seed_id: str, qdrant_host: str, qdrant_port: int) -> list[str]:
    """Drop a global seed ontology graph via ``olaf drop-seed <seed_id>``.

    Seeds carry no Qdrant-side data (no concepts collection is ever created
    for one), so unlike :func:`delete_ontology_for_collection` this only
    touches Oxigraph. ``qdrant_host``/``qdrant_port`` are only needed to
    render a valid ``config.toml`` for the ``olaf`` subprocess — the
    drop-seed code path never actually connects to Qdrant.
    """
    with _scratch_config_dir(
        qdrant_host=qdrant_host, qdrant_port=qdrant_port, qdrant_collection="_unused_",
        ontology_id="_scratch_", ontology_name="scratch",
    ) as config_dir:
        _run_olaf(["drop-seed", seed_id], config_dir)
    return [f"Seed 'urn:olaf:seed:{seed_id}' dropped."]
