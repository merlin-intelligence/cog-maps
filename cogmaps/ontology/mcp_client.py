"""Spawns OLAF as a local stdio MCP subprocess and wraps the client session.

OLAF (https://github.com/merlin-intelligence/olaf) defaults to stdio
transport (no ``MCP_TRANSPORT=sse`` env set) — so rather than deploying a
standalone OLAF server, we launch one subprocess per ontology-building job,
reading the ``config.toml`` written to ``cwd`` by
:mod:`cogmaps.ontology.olaf_config`. The subprocess goes through
:mod:`cogmaps.ontology.olaf_launcher` instead of the ``olaf`` console script,
so OLAF's fastembed can load cog-maps' embedding model.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cogmaps.config import PROJECT_ROOT


def launcher_params(config_dir: str) -> StdioServerParameters:
    """Run :mod:`cogmaps.ontology.olaf_launcher` with this interpreter, from ``config_dir``."""
    env = os.environ.copy()
    # cwd is a temp dir, so cogmaps must be importable even when it isn't pip-installed.
    paths = [str(PROJECT_ROOT), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return StdioServerParameters(
        command=sys.executable, args=["-m", "cogmaps.ontology.olaf_launcher"], cwd=config_dir, env=env,
    )


@asynccontextmanager
async def olaf_session(config_dir: str):
    """Async context manager yielding an initialized :class:`ClientSession` for OLAF.

    Spawns OLAF through :func:`launcher_params` with ``cwd=config_dir``, so
    it picks up the ``config.toml`` written there for this job.
    """
    params = launcher_params(config_dir)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session
