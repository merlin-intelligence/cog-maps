"""Runs OLAF's administrative CLI commands as short-lived subprocesses.

OLAF keeps some operations out of its MCP tools — ``olaf drop``,
``drop-seed``, ``load-seed`` (local files, internal URLs), ``reindex``,
``index-seed`` (see https://github.com/merlin-intelligence/olaf, "CLI
commands"). Like a build job's server, each command reads the
``config.toml`` of its working directory, and goes through
:mod:`cogmaps.ontology.olaf_launcher` rather than the ``olaf`` console script
so that the commands which embed (``load-seed``, ``reindex``, ``index-seed``)
can load cog-maps' embedding model.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

from cogmaps.config import oxigraph_url as cfg_oxigraph_url
from cogmaps.config import qdrant_api_key as cfg_qdrant_api_key
from cogmaps.ontology.mcp_client import launcher_params
from cogmaps.ontology.olaf_config import build_config_toml

DEFAULT_TIMEOUT = 60

# Every CLI command bootstraps the config's ontology (an owl:Ontology
# declaration in urn:olaf:{id}) before running: commands that are not about
# an ontology get this throwaway one, dropped again afterwards.
SCRATCH_ONTOLOGY_ID = "_scratch_"


def run_olaf(args: list[str], config_dir: str, *, timeout: float = DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
    """Run ``olaf <args>`` from ``config_dir``; raises ``RuntimeError`` with OLAF's output if it fails."""
    params = launcher_params(config_dir)
    # OLAF's output is shown on the pages: without library warnings (insecure API key, client version…).
    env = {**params.env, "PYTHONWARNINGS": "ignore"}
    result = subprocess.run(
        [params.command, *params.args, *args], cwd=config_dir, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"'olaf {' '.join(args)}' failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def write_config(
    config_dir: str, *, qdrant_url: str, qdrant_collection: str, ontology_id: str, ontology_name: str,
) -> None:
    """Write the OLAF ``config.toml`` for ``ontology_id`` into ``config_dir``."""
    config_toml = build_config_toml(
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        qdrant_api_key=cfg_qdrant_api_key(),
        oxigraph_url=cfg_oxigraph_url(),
        ontology_id=ontology_id,
        ontology_name=ontology_name,
    )
    with open(os.path.join(config_dir, "config.toml"), "w", encoding="utf-8") as f:
        f.write(config_toml)


@contextmanager
def scratch_config_dir(
    *,
    qdrant_url: str,
    qdrant_collection: str = "_unused_",
    ontology_id: str = SCRATCH_ONTOLOGY_ID,
    ontology_name: str = "scratch",
) -> Iterator[str]:
    """A temp dir holding a config.toml for one-off ``olaf`` CLI invocations."""
    with tempfile.TemporaryDirectory(prefix=f"cogmaps_olaf_{ontology_id}_") as config_dir:
        write_config(
            config_dir, qdrant_url=qdrant_url, qdrant_collection=qdrant_collection,
            ontology_id=ontology_id, ontology_name=ontology_name,
        )
        try:
            yield config_dir
        finally:
            if ontology_id == SCRATCH_ONTOLOGY_ID:
                _drop_scratch_graph()


def _drop_scratch_graph() -> None:
    """Remove the scratch ontology a command bootstrapped (HTTP graph store DELETE)."""
    endpoint = cfg_oxigraph_url().rstrip("/")
    req = urllib.request.Request(f"{endpoint}/store?graph=urn:olaf:{SCRATCH_ONTOLOGY_ID}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:  # noqa: BLE001 — 404 when absent, or Oxigraph down: nothing to clean
        pass
