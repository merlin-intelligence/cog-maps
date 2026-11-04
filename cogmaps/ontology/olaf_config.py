"""Builds the OLAF server ``config.toml`` for one ontology-building job.

OLAF (https://github.com/merlin-intelligence/olaf) reads a single
``config.toml``/``olaf.toml`` from its working directory (see
``olaf.config.Config.load``) — there is no per-call override of which
Qdrant collection it reads chunks from. So instead of running one long-lived
OLAF server, we spawn a fresh ``olaf`` subprocess per job (see
:mod:`cogmaps.ontology.mcp_client`), with a config file generated here and
written to that job's temp directory.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from rdflib import OWL, RDF, Graph

from cogmaps.config import EMBEDDING_MODEL_NAME, USER_DATA_DIR
from cogmaps.config import oxigraph_url as default_oxigraph_url

logger = logging.getLogger(__name__)

_ONTOLOGY_TYPES = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual)

_ONTOLOGY_ID_RE = re.compile(r"[^a-z0-9_]+")

# Matches the [qdrant].index_prefix rendered by build_config_toml below — OLAF's
# semantic index lives in the Qdrant collections f"{INDEX_PREFIX}_{ontology_id}"
# (one per ontology) and f"{INDEX_PREFIX}_seed_{seed_id}" (one per seed).
INDEX_PREFIX = "olaf_index"

# Prefix of the per-ontology concepts collections written by OLAF before it
# renamed them to INDEX_PREFIX — left behind by ontologies built back then,
# replaced on their next job (see cogmaps.ontology.seeds.ensure_ontology_index).
LEGACY_CONCEPTS_PREFIX = "olaf_concepts"

# Separates a collection's default ontology id from the name of its other ontologies.
ONTOLOGY_NAME_SEP = "__"

# OLAF's former chunk processing status fields, shared by every ontology of a
# collection. OLAF now suffixes them with the ontology id (each ontology of a
# collection has its own progress); ontologies built before still have these.
SHARED_STATUS_FIELDS = ("olaf_status", "olaf_processed_at")


def chunk_status_field(ontology_id: str) -> str:
    """OLAF's per-ontology status payload field (ontology ids need no escaping)."""
    return f"{SHARED_STATUS_FIELDS[0]}_{ontology_id}"


def index_collection_for(ontology_id: str) -> str:
    """The Qdrant collection holding the semantic index of an ontology."""
    return f"{INDEX_PREFIX}_{ontology_id}"


def seed_index_collection_for(seed_id: str) -> str:
    """The Qdrant collection holding the semantic index of a seed."""
    return f"{INDEX_PREFIX}_seed_{seed_id}"


def sanitize_ontology_id(collection: str) -> str:
    """Turn a Qdrant collection name into a valid OLAF ``ontology_id``.

    Used as the named-graph suffix (``urn:olaf:{id}``) and the Qdrant
    index-collection suffix (``olaf_index_{id}``) — keep it to
    lowercase alphanumerics/underscores.
    """
    slug = _ONTOLOGY_ID_RE.sub("_", collection.lower()).strip("_")
    return slug or "main"


def ontology_id_for(collection: str, name: str = "") -> str:
    """The ``ontology_id`` of one of a collection's ontologies.

    A collection can have several: its default one (``name`` empty), whose id is
    the sanitized collection name, and named ones, ``{default id}__{sanitized name}``
    — derived from the collection, so they stay within its ownership boundary.
    """
    default = sanitize_ontology_id(collection)
    slug = _ONTOLOGY_ID_RE.sub("_", name.lower()).strip("_")
    return f"{default}{ONTOLOGY_NAME_SEP}{slug}" if slug else default


def ontology_name_of(collection: str, ontology_id: str) -> str:
    """The name part of one of the collection's ontology ids ("" for the default ontology)."""
    prefix = sanitize_ontology_id(collection) + ONTOLOGY_NAME_SEP
    return ontology_id[len(prefix):] if ontology_id.startswith(prefix) else ""


def ontology_display_name(collection: str, ontology_id: str) -> str:
    """The ``name`` OLAF records for the ontology: the collection for its default one
    (as before collections had several), else the collection and the ontology's name."""
    name = ontology_name_of(collection, ontology_id)
    return f"{collection} — {name}" if name else collection


def collection_ontology_ids(collection: str, graphs) -> list[str]:
    """The collection's ontologies among the named graphs ``graphs`` (``urn:olaf:{id}``),
    the default one first, then the named ones by name."""
    default = sanitize_ontology_id(collection)
    ids = {g[len("urn:olaf:"):] for g in graphs if g.startswith("urn:olaf:")}
    named = sorted(i for i in ids if i.startswith(default + ONTOLOGY_NAME_SEP))
    return ([default] if default in ids else []) + named


def _local_name(collection: str, ontology_id: str | None) -> str:
    """The name of an ontology's local files: the collection's for its default ontology
    (as before collections had several), the ontology id for the others."""
    if ontology_id is None or ontology_id == sanitize_ontology_id(collection):
        return collection
    return ontology_id


def ttl_path_for(collection: str, ontology_id: str | None = None) -> str:
    """Where the exported Turtle of a collection's ontology (its default one when
    ``ontology_id`` is None) is persisted.

    Keyed by the (already user/public-namespaced) Qdrant collection name, not
    a separate user field — matching how the rest of the app treats the
    collection name as the ownership boundary.
    """
    return str(USER_DATA_DIR / "ontology" / f"{_local_name(collection, ontology_id)}.ttl")


def alignment_path_for(collection: str, ontology_id: str) -> str:
    """Where the alignment (SSSOM TSV) of a merged ontology's two sources is persisted."""
    return str(USER_DATA_DIR / "ontology" / f"{_local_name(collection, ontology_id)}.alignment.sssom.tsv")


def backup_dir_for(collection: str, ontology_id: str | None = None) -> Path:
    """Where the reasoning agent's pre-change Turtle backups of a collection's ontology go."""
    return USER_DATA_DIR / "ontology" / "backups" / _local_name(collection, ontology_id)


def save_backup(collection: str, ontology_id: str | None, ttl: str) -> Path:
    """Write ``ttl`` as a new timestamped backup of the collection's ontology."""
    directory = backup_dir_for(collection, ontology_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}.ttl"
    path.write_text(ttl, encoding="utf-8")
    return path


def list_backups(collection: str, ontology_id: str | None = None) -> list[Path]:
    """The ontology's backups, most recent first."""
    directory = backup_dir_for(collection, ontology_id)
    return sorted(directory.glob("*.ttl"), reverse=True) if directory.is_dir() else []


def restore_oxigraph_graph(
    ontology_id: str, ttl: str, oxigraph_url: str | None = None, *, timeout: float = 30,
) -> None:
    """Replace the ontology's named graph in Oxigraph with ``ttl`` (HTTP graph store PUT).

    Only the graph is restored: OLAF's semantic index (the ``olaf_index_*``
    Qdrant collection) keeps what the reasoning run left there, e.g. without
    the concepts it merged away.
    """
    endpoint = (oxigraph_url or default_oxigraph_url()).rstrip("/")
    req = urllib.request.Request(
        f"{endpoint}/store?graph=urn:olaf:{ontology_id}",
        data=ttl.encode("utf-8"),
        headers={"Content-Type": "text/turtle"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=timeout):
        pass


def has_ontology_content(ttl: str | None) -> bool:
    """True if ``ttl`` parses as Turtle and declares at least one class, property or individual.

    Used to decide whether an export is worth persisting, instead of
    substring checks that miss prefix-less or differently-formatted Turtle.
    """
    if not ttl or not ttl.strip():
        return False
    g = Graph()
    try:
        g.parse(data=ttl, format="turtle")
    except Exception as e:  # noqa: BLE001
        logger.warning("Ontology export is not valid Turtle: %s", e)
        return False
    return any(next(g.subjects(RDF.type, t), None) is not None for t in _ONTOLOGY_TYPES)


def export_oxigraph_ontology_ttl(
    ontology_id: str, oxigraph_url: str | None = None, *, timeout: float = 10,
) -> str | None:
    """Fetch the Turtle representation of an ontology's named graph directly from Oxigraph.

    Bypasses seed graphs and guarantees returning the current state in the graph store.
    Returns None if the graph is missing or Oxigraph is unreachable.
    """
    endpoint = (oxigraph_url or default_oxigraph_url()).rstrip("/")
    url = f"{endpoint}/store?graph=urn:olaf:{ontology_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/turtle"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("No Oxigraph graph urn:olaf:%s yet", ontology_id)
        else:
            logger.warning("Oxigraph export of urn:olaf:%s failed: HTTP %s", ontology_id, e.code)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Oxigraph export of urn:olaf:%s failed: %s", ontology_id, e)
        return None
    return data if data.strip() else None


def build_config_toml(
    *,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_api_key: str | None,
    oxigraph_url: str,
    ontology_id: str,
    ontology_name: str,
    base_uri: str = "http://olaf.local/ontology#",
) -> str:
    """Render OLAF's ``config.toml`` content for one job.

    Field mapping matches CogMaps' actual Qdrant payload
    (:mod:`cogmaps.qdrant.store`): ``filename`` for ``doc_id`` and
    ``chunk_number`` for ``chunk_index``.

    ``qdrant_api_key`` requires a patched OLAF that reads ``[qdrant].api_key``
    and passes it to ``QdrantClient(url=..., api_key=...)`` — CogMaps' own
    Qdrant instance refuses unauthenticated requests once ``QDRANT_API_KEY``
    is set (see ``docker-compose.yml``), and upstream OLAF has no api_key
    field at all. Omitted from the rendered file when unset.
    """
    api_key_line = f'api_key             = "{qdrant_api_key}"\n' if qdrant_api_key else ""
    return f'''[qdrant]
url                 = "{qdrant_url}"
collection          = "{qdrant_collection}"
{api_key_line}index_prefix        = "{INDEX_PREFIX}"

[qdrant.field_mapping]
text        = "text"
doc_id      = "filename"
chunk_index = "chunk_number"

[oxigraph]
url = "{oxigraph_url}"

[ontology]
base_uri    = "{base_uri}"
name        = "{ontology_name}"
ontology_id = "{ontology_id}"

[embedding]
# Not in fastembed's catalog — registered by cogmaps.ontology.olaf_launcher.
model   = "{EMBEDDING_MODEL_NAME}"
enabled = true
'''
