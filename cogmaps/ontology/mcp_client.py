"""Spawns OLAF as a local stdio MCP subprocess and wraps the client session.

OLAF (https://github.com/merlin-intelligence/olaf) defaults to stdio
transport when run as the ``olaf`` console script (no ``MCP_TRANSPORT=sse``
env set) — so rather than deploying a standalone OLAF server, we launch one
subprocess per ontology-building job, reading the ``config.toml`` written
to ``cwd`` by :mod:`cogmaps.ontology.olaf_config`.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def olaf_session(config_dir: str):
    """Async context manager yielding an initialized :class:`ClientSession` for OLAF.

    Spawns ``olaf`` (the console script installed from the OLAF package)
    with ``cwd=config_dir``, so it picks up the ``config.toml`` written
    there for this job.
    """
    params = StdioServerParameters(command="olaf", args=[], cwd=config_dir)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session
