"""Daemon thread that processes ontology-building jobs from an in-memory queue.

Mirrors :class:`cogmaps.jobs.runner.JobRunner`'s shape (single worker thread,
sequential processing) — here sequential also avoids concurrent writers to
the same Oxigraph store. Each job spawns its own short-lived ``olaf``
subprocess (see :mod:`cogmaps.ontology.mcp_client`) with a config file
generated for that job's collection/ontology.
"""
from __future__ import annotations

import asyncio
import os
import queue
import shutil
import tempfile
import threading
import traceback

from cogmaps.config import nebius_api_key, oxigraph_url, qdrant_api_key
from cogmaps.ontology.agent import run_build
from cogmaps.ontology.mcp_client import olaf_session
from cogmaps.ontology.olaf_config import build_config_toml, ttl_path_for
from cogmaps.ontology.store import OntologyJobStore
from cogmaps.rag.llm_clients import resolve_nebius_endpoint

__all__ = ["OntologyJobRunner", "ttl_path_for"]


class OntologyJobRunner:
    """Single-worker background runner for ontology-building jobs."""

    def __init__(self, store: OntologyJobStore, qdrant_host: str, qdrant_port: int) -> None:
        self.store = store
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="cogmaps-ontology"
        )
        self._thread.start()

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    # ── private ──────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                asyncio.run(self._run(job_id))
            except Exception:  # noqa: BLE001
                self.store.set_status(job_id, "failed")
                self.store.append_log(job_id, f"Unexpected runner error:\n{traceback.format_exc()}")
            finally:
                self._queue.task_done()

    async def _run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return

        self.store.set_status(job_id, "running")

        def log(message: str) -> None:
            self.store.append_log(job_id, message)

        def set_progress(current: int, total: int) -> None:
            self.store.set_progress(job_id, current, total)

        collection = job["collection"]
        config_dir = tempfile.mkdtemp(prefix=f"cogmaps_olaf_{job_id}_")
        try:
            config_toml = build_config_toml(
                qdrant_url=f"http://{self.qdrant_host}:{self.qdrant_port}",
                qdrant_collection=collection,
                qdrant_api_key=qdrant_api_key(),
                oxigraph_url=oxigraph_url(),
                ontology_id=job["ontology_id"],
                ontology_name=collection,
            )
            with open(os.path.join(config_dir, "config.toml"), "w", encoding="utf-8") as f:
                f.write(config_toml)

            # litellm's api_base is the endpoint root (no trailing path), same
            # convention as OpenAI's own api_base — strip the fixed suffix off
            # the full chat-completions URL NebiusClient uses.
            chat_url, _vendor = resolve_nebius_endpoint(job["llm_model"])
            api_base = chat_url.removesuffix("/chat/completions")

            def export_cb(turtle: str) -> None:
                out_path = ttl_path_for(collection)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(turtle)

            log(f"Starting OLAF (collection={collection!r}, ontology_id={job['ontology_id']!r})")
            async with olaf_session(config_dir) as session:
                await run_build(
                    session,
                    model=job["llm_model"],
                    api_base=api_base,
                    api_key=nebius_api_key(),
                    doc_filenames=job["doc_filenames"],
                    log=log,
                    set_progress=set_progress,
                    export_cb=export_cb,
                )
            self.store.set_status(job_id, "done")
            log("Ontology build complete.")
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, "failed")
            log(f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)
