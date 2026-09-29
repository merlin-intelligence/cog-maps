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

import re

from cogmaps.config import USER_DATA_DIR

_ONTOLOGY_ID_RE = re.compile(r"[^a-z0-9_]+")

# Matches the [qdrant].concepts_collection prefix rendered by build_config_toml
# below — the actual Qdrant collection OLAF's EmbeddingService writes to is
# f"{CONCEPTS_COLLECTION_PREFIX}_{ontology_id}". Shared with cleanup.py so
# deleting an ontology's concepts collection can't drift out of sync.
CONCEPTS_COLLECTION_PREFIX = "olaf_concepts"


def sanitize_ontology_id(collection: str) -> str:
    """Turn a Qdrant collection name into a valid OLAF ``ontology_id``.

    Used as the named-graph suffix (``urn:olaf:{id}``) and the Qdrant
    concepts-collection suffix (``olaf_concepts_{id}``) — keep it to
    lowercase alphanumerics/underscores.
    """
    slug = _ONTOLOGY_ID_RE.sub("_", collection.lower()).strip("_")
    return slug or "main"


def ttl_path_for(collection: str) -> str:
    """Where the exported Turtle for a Qdrant collection's ontology is persisted.

    Keyed by the (already user/public-namespaced) Qdrant collection name, not
    a separate user field — matching how the rest of the app treats the
    collection name as the ownership boundary.
    """
    return str(USER_DATA_DIR / "ontology" / f"{collection}.ttl")


def export_oxigraph_ontology_ttl(ontology_id: str, oxigraph_url: str | None = None) -> str | None:
    """Fetch the Turtle representation of an ontology's named graph directly from Oxigraph.

    Bypasses seed graphs and guarantees returning the current state in the graph store.
    """
    import urllib.request
    from cogmaps.config import oxigraph_url as default_oxigraph_url

    endpoint = (oxigraph_url or default_oxigraph_url()).rstrip("/")
    url = f"{endpoint}/store?graph=urn:olaf:{ontology_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/turtle"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            if data and data.strip():
                return data
    except Exception:
        pass
    return None


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
{api_key_line}concepts_collection = "{CONCEPTS_COLLECTION_PREFIX}"

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
model   = "intfloat/multilingual-e5-base"
enabled = true
'''
