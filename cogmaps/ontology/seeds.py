"""Seed ontologies (global reference ontologies) and OLAF's semantic index.

Seeds live in Oxigraph as ``urn:olaf:seed:{id}`` named graphs, shared across
every ontology, and are indexed for semantic search in the Qdrant collection
``olaf_index_seed_{id}`` (see OLAF's README, "Seeds"). They are loaded from
the Ontology Explorer page with OLAF's ``load-seed`` CLI command
(:mod:`cogmaps.ontology.olaf_cli`): unlike the ``seed_load`` MCP tool, it
reads local files and URLs on internal hosts — the page restricts it to admins.

A build job then selects the seeds its agent may reuse terms from
(``seed_select``, see :mod:`cogmaps.ontology.agent`).
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient

from cogmaps.config import oxigraph_url as cfg_oxigraph_url
from cogmaps.ontology.olaf_cli import run_olaf, scratch_config_dir
from cogmaps.ontology.olaf_config import (
    LEGACY_CONCEPTS_PREFIX,
    index_collection_for,
    seed_index_collection_for,
)
from cogmaps.ontology.oxigraph_client import is_reachable, run_query

SEED_GRAPH_PREFIX = "urn:olaf:seed:"

# OLAF's ``load-seed --format`` choices (``olaf.ontology.RDF_FORMATS``).
SEED_FORMATS = ("ttl", "xml", "nt", "n3", "jsonld")

# Loading a large seed (e.g. FIBO) and embedding its entities takes minutes.
_LOAD_TIMEOUT = 30 * 60

_SEEDS_QUERY = f"""
SELECT ?g (COUNT(*) AS ?triples) WHERE {{
    GRAPH ?g {{ ?s ?p ?o }}
    FILTER(STRSTARTS(STR(?g), "{SEED_GRAPH_PREFIX}"))
}} GROUP BY ?g ORDER BY ?g
"""


@dataclass
class SeedInfo:
    seed_id: str
    triples: int
    indexed_entities: int | None  # points in its Qdrant index, None if it has none


def list_seeds(qdrant: QdrantClient | None = None) -> list[SeedInfo]:
    """Seeds loaded in Oxigraph, with their size — and, given ``qdrant``, how many entities are indexed.

    Returns an empty list when Oxigraph is unreachable.
    """
    url = cfg_oxigraph_url()
    if not is_reachable(url):
        return []
    _kind, payload = run_query(url, _SEEDS_QUERY)
    seeds = []
    for row in payload.get("results", {}).get("bindings", []):
        seed_id = row["g"]["value"][len(SEED_GRAPH_PREFIX):]
        indexed = None
        if qdrant is not None:
            collection = seed_index_collection_for(seed_id)
            if qdrant.collection_exists(collection):
                indexed = qdrant.count(collection).count
        seeds.append(SeedInfo(seed_id, int(row["triples"]["value"]), indexed))
    return seeds


def _seed_args(seed_id: str | None, fmt: str | None) -> list[str]:
    args = []
    if seed_id:
        args += ["--id", seed_id]
    if fmt:
        args += ["--format", fmt]
    return args


def load_seed_from_url(url: str, *, qdrant_url: str, seed_id: str | None = None, fmt: str | None = None) -> str:
    """Load (or replace, for an existing ``seed_id``) a seed from ``url`` and index it. Returns OLAF's report."""
    with scratch_config_dir(qdrant_url=qdrant_url) as config_dir:
        result = run_olaf(["load-seed", url, *_seed_args(seed_id, fmt)], config_dir, timeout=_LOAD_TIMEOUT)
    return (result.stdout + result.stderr).strip()


def load_seed_from_file(
    data: bytes, filename: str, *, qdrant_url: str, seed_id: str | None = None, fmt: str | None = None,
) -> str:
    """Load (or replace) a seed from an uploaded file and index it. Returns OLAF's report.

    The file keeps its name: OLAF guesses the format from its extension and,
    without ``seed_id``, derives the id from it.
    """
    with tempfile.TemporaryDirectory(prefix="cogmaps_seed_") as tmp_dir:
        path = os.path.join(tmp_dir, os.path.basename(filename) or "seed")
        with open(path, "wb") as f:
            f.write(data)
        with scratch_config_dir(qdrant_url=qdrant_url) as config_dir:
            result = run_olaf(["load-seed", path, *_seed_args(seed_id, fmt)], config_dir, timeout=_LOAD_TIMEOUT)
    return (result.stdout + result.stderr).strip()


def index_seed(seed_id: str, *, qdrant_url: str) -> str:
    """(Re)build a seed's Qdrant index from Oxigraph. Returns OLAF's report."""
    with scratch_config_dir(qdrant_url=qdrant_url) as config_dir:
        result = run_olaf(["index-seed", seed_id], config_dir, timeout=_LOAD_TIMEOUT)
    return (result.stdout + result.stderr).strip()


def _has_entities(ontology_id: str) -> bool:
    query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    ASK {{ GRAPH <urn:olaf:{ontology_id}> {{
        ?e a ?t . VALUES ?t {{ owl:Class owl:NamedIndividual owl:ObjectProperty owl:DatatypeProperty }}
    }} }}
    """
    _kind, payload = run_query(cfg_oxigraph_url(), query)
    return bool(payload.get("boolean"))


def ensure_ontology_index(
    ontology_id: str, config_dir: str, qdrant: QdrantClient, log: Callable[[str], None],
) -> None:
    """Build the ontology's semantic index if it is missing, and drop its legacy one.

    Ontologies built before OLAF renamed its index have an
    ``olaf_concepts_{id}`` collection (classes only) and no
    ``olaf_index_{id}``; OLAF's searches only read the latter. ``config_dir``
    holds the job's OLAF config, for ``olaf reindex``.
    """
    index = index_collection_for(ontology_id)
    legacy = f"{LEGACY_CONCEPTS_PREFIX}_{ontology_id}"
    has_legacy = qdrant.collection_exists(legacy)
    # A search session creates the index collection empty when it is missing.
    missing = not qdrant.collection_exists(index) or qdrant.count(index).count == 0
    if (has_legacy or missing) and _has_entities(ontology_id):
        log(f"Semantic index '{index}' missing — rebuilding it from Oxigraph (olaf reindex)...")
        result = run_olaf(["reindex", ontology_id], config_dir, timeout=_LOAD_TIMEOUT)
        log(result.stdout.strip() or "Semantic index rebuilt.")
    if has_legacy:
        qdrant.delete_collection(legacy)
        log(f"Legacy concepts collection '{legacy}' deleted.")
