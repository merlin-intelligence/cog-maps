"""Daemon thread that processes ontology jobs (builds and reasoning runs) from an in-memory queue.

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
from collections.abc import Callable

from qdrant_client import models

from cogmaps.config import domain_discovery_model, oxigraph_url, qdrant_api_key, scaleway_api_key
from cogmaps.core.llm_impacts import record_llm_impacts
from cogmaps.ontology.agent import run_build
from cogmaps.ontology.domain_discovery import discover_domain, load_blueprint
from cogmaps.ontology.mcp_client import olaf_session
from cogmaps.ontology.olaf_config import (
    build_config_toml,
    export_oxigraph_ontology_ttl,
    has_ontology_content,
    save_backup,
    ttl_path_for,
)
from cogmaps.ontology.reasoning_agent import run_reasoning
from cogmaps.ontology.store import OntologyJobStore
from cogmaps.qdrant.store import QdrantStore

__all__ = ["OntologyJobRunner", "final_export", "ttl_path_for"]


# OLAF's chunk-status payload field (``olaf.chunks._STATUS_FIELD``).
OLAF_STATUS_FIELD = "olaf_status"


def init_pending_chunks(client, collection: str, doc_filenames: list[str]) -> None:
    """Tag the requested documents' never-processed chunks as ``olaf_status="pending"``.

    OLAF only writes this field when a chunk is marked processed, but its
    ``chunk_list(status="pending")`` filters on the field's value — so on a
    fresh collection it returns nothing and the agent believes there is no
    work left. Chunks already marked processed are left untouched.
    """
    must: list = [models.IsEmptyCondition(is_empty=models.PayloadField(key=OLAF_STATUS_FIELD))]
    if doc_filenames:
        must.append(models.FieldCondition(key="filename", match=models.MatchAny(any=list(doc_filenames))))
    client.set_payload(
        collection_name=collection,
        payload={OLAF_STATUS_FIELD: "pending"},
        points=models.FilterSelector(filter=models.Filter(must=must)),
        wait=True,
    )


async def final_export(
    session,
    ontology_id: str,
    *,
    export_cb: Callable[[str], None],
    log: Callable[[str], None],
) -> None:
    """Guaranteed final export of the active ontology (excluding global seeds).

    Asks OLAF for the export first; if that fails or looks empty, falls back
    to reading the named graph straight from Oxigraph.
    """
    try:
        res = await session.call_tool("ontology_export", {"include_seeds": False})
        content = res.content[0].text if res.content else ""
        if has_ontology_content(content):
            export_cb(content)
            log("Final ontology exported successfully.")
            return
        raise ValueError("Session export returned empty or incomplete content")
    except Exception as ex:  # noqa: BLE001
        log(f"Final session export fallback: querying Oxigraph directly ({ex})...")
        direct_ttl = export_oxigraph_ontology_ttl(ontology_id, oxigraph_url())
        if direct_ttl:
            export_cb(direct_ttl)
            log("Final ontology exported via direct Oxigraph connection.")


class OntologyJobRunner:
    """Single-worker background runner for ontology jobs.

    Builds and reasoning runs share the one worker, so they never write to the
    same Oxigraph graph at the same time.
    """

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
                # Every LLM request of the build (discovery + agent loop) is kept
                # with the job, for the page's impact footer.
                with record_llm_impacts(lambda call, job_id=job_id: self.store.append_llm_call(job_id, call.to_dict())):
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
        if job.get("job_type") == "reasoning":
            await self._run_reasoning(job, log, set_progress)
            return
        config_dir = tempfile.mkdtemp(prefix=f"cogmaps_olaf_{job_id}_")
        try:
            self._write_config(config_dir, job)
            export_cb = _ttl_writer(collection)

            # ── Domain discovery (optional, opted into per job) ──
            blueprint = None
            if job["use_domain_discovery"]:
                log(f"Domain discovery: resolving domain blueprint for collection {collection!r}...")
                blueprint = load_blueprint(collection)
                if blueprint is None:
                    discovery_model = domain_discovery_model()
                    log(f"Domain discovery: no cached blueprint found, synthesizing one with {discovery_model}...")
                    try:
                        store = QdrantStore(self.qdrant_host, self.qdrant_port)
                        blueprint = discover_domain(store, collection, model=discovery_model)
                        log(
                            f"Domain discovery: completed -> '{blueprint.inferred_domain}' "
                            f"({blueprint.epistemological_nature}) with {len(blueprint.pillars)} taxonomical pillars."
                        )
                    except Exception as e:  # noqa: BLE001
                        log(f"Domain discovery warning: failed ({e}), proceeding with default prompt.")
                        blueprint = None
                else:
                    log(f"Domain discovery: loaded cached blueprint -> '{blueprint.inferred_domain}' ({len(blueprint.pillars)} pillars).")
            else:
                log("Domain discovery disabled for this build — using the default prompt.")

            init_pending_chunks(
                QdrantStore(self.qdrant_host, self.qdrant_port).client, collection, job["doc_filenames"]
            )

            log(f"Starting OLAF (collection={collection!r}, ontology_id={job['ontology_id']!r})")
            async with olaf_session(config_dir) as session:
                await run_build(
                    session,
                    model=job["llm_model"],
                    api_key=scaleway_api_key(),
                    doc_filenames=job["doc_filenames"],
                    log=log,
                    set_progress=set_progress,
                    export_cb=export_cb,
                    blueprint=blueprint,
                )
                await final_export(session, job["ontology_id"], export_cb=export_cb, log=log)
            self.store.set_status(job_id, "done")
            log("Ontology build complete.")
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, "failed")
            log(f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

    async def _run_reasoning(self, job: dict, log: Callable[[str], None], set_progress) -> None:
        """Curate the ontology with the reasoning agent: dedup, repair, enrich, infer."""
        job_id, collection, options = job["id"], job["collection"], job["options"]
        config_dir = tempfile.mkdtemp(prefix=f"cogmaps_olaf_{job_id}_")
        try:
            self._write_config(config_dir, job)

            def backup_cb(turtle: str) -> None:
                log(f"Backup of the ontology before any change: {save_backup(collection, turtle)}")

            log(f"Starting OLAF reasoning (collection={collection!r}, ontology_id={job['ontology_id']!r})")
            async with olaf_session(config_dir) as session:
                summary = await run_reasoning(
                    session,
                    model=job["llm_model"],
                    api_key=scaleway_api_key(),
                    ontology_id=job["ontology_id"],
                    log=log,
                    set_progress=set_progress,
                    backup_cb=backup_cb,
                    export_cb=_ttl_writer(collection),
                    max_rounds=options.get("max_rounds", 3),
                    fix_orphans=options.get("fix_orphans", True),
                    dedup=options.get("dedup", True),
                    enrich_disjointness=options.get("enrich_disjointness", True),
                    infer=options.get("infer", True),
                    review_inferences=options.get("review_inferences", True),
                )
            self.store.set_result(job_id, summary)
            self.store.set_status(job_id, "done")
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, "failed")
            log(f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

    def _write_config(self, config_dir: str, job: dict) -> None:
        """Write the OLAF ``config.toml`` for this job's collection/ontology into ``config_dir``."""
        config_toml = build_config_toml(
            qdrant_url=f"http://{self.qdrant_host}:{self.qdrant_port}",
            qdrant_collection=job["collection"],
            qdrant_api_key=qdrant_api_key(),
            oxigraph_url=oxigraph_url(),
            ontology_id=job["ontology_id"],
            ontology_name=job["collection"],
        )
        with open(os.path.join(config_dir, "config.toml"), "w", encoding="utf-8") as f:
            f.write(config_toml)


def _ttl_writer(collection: str) -> Callable[[str], None]:
    """Callback persisting an exported Turtle to the collection's on-disk fallback copy."""
    def export_cb(turtle: str) -> None:
        out_path = ttl_path_for(collection)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(turtle)
    return export_cb
