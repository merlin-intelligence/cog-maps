"""Daemon thread that processes ontology jobs (builds, merges and reasoning runs) from an in-memory queue.

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

from cogmaps.config import domain_discovery_model, oxigraph_url, scaleway_api_key
from cogmaps.core.llm_impacts import record_llm_impacts
from cogmaps.ontology.agent import run_build
from cogmaps.ontology.domain_discovery import (
    DomainBlueprint,
    discover_domain,
    load_blueprint,
    load_manual_guidelines,
)
from cogmaps.ontology.mcp_client import olaf_session
from cogmaps.ontology.merging_agent import run_merge
from cogmaps.ontology.olaf_cli import write_config
from cogmaps.ontology.olaf_config import (
    SHARED_STATUS_FIELDS,
    alignment_path_for,
    chunk_status_field,
    export_oxigraph_ontology_ttl,
    has_ontology_content,
    ontology_display_name,
    sanitize_ontology_id,
    save_backup,
    ttl_path_for,
)
from cogmaps.ontology.reasoning_agent import run_reasoning
from cogmaps.ontology.seeds import ensure_ontology_index
from cogmaps.ontology.store import OntologyJobStore
from cogmaps.qdrant.store import QdrantStore

__all__ = ["OntologyJobRunner", "build_domain_mode", "build_seed_ids", "final_export", "ttl_path_for"]


def migrate_shared_chunk_status(client, collection: str) -> int:
    """Hand the collection's shared chunk status over to its default ontology.

    Ontologies built before collections had several were tracked with OLAF's
    shared ``olaf_status`` field, which only the default ontology can have
    written. Its processed chunks are marked processed for it under its own
    per-ontology field, then the shared fields are removed — so the next
    build of any of the collection's ontologies sees the right progress, and
    this runs once. Returns the number of chunks handed over.
    """
    status_field = SHARED_STATUS_FIELDS[0]
    with_status = models.Filter(must_not=[
        models.IsEmptyCondition(is_empty=models.PayloadField(key=status_field)),
    ])
    if not client.count(collection_name=collection, count_filter=with_status).count:
        return 0
    processed = models.Filter(must=[
        models.FieldCondition(key=status_field, match=models.MatchValue(value="processed")),
    ])
    count = client.count(collection_name=collection, count_filter=processed).count
    if count:
        client.set_payload(
            collection_name=collection,
            payload={chunk_status_field(sanitize_ontology_id(collection)): "processed"},
            points=models.FilterSelector(filter=processed),
            wait=True,
        )
    # Also drops the "pending" values cog-maps wrote for older OLAF versions.
    client.delete_payload(
        collection_name=collection,
        keys=list(SHARED_STATUS_FIELDS),
        points=models.FilterSelector(filter=with_status),
        wait=True,
    )
    return count


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


def job_domain_mode(job: dict) -> str:
    """Where a build's domain guidance comes from: "none", "llm" (discovered blueprint) or
    "manual" (hand-written guidelines). Jobs from before the choice only had the LLM option."""
    return job["options"].get("domain_mode") or ("llm" if job["use_domain_discovery"] else "none")


def build_domain_mode(store: OntologyJobStore, collection: str, ontology_id: str) -> str:
    """The domain guidance mode of the ontology's latest build ("none" if never built)."""
    job = store.latest_job_for(collection, "build", ontology_id)
    return job_domain_mode(job) if job else "none"


def build_seed_ids(store: OntologyJobStore, collection: str, ontology_id: str) -> list[str]:
    """The seeds selected for the ontology's latest build — those it reuses terms from. For a
    merged ontology, those of its two sources."""
    job = store.latest_job_for(collection, "build", ontology_id) or store.latest_job_for(
        collection, "merge", ontology_id,
    )
    return job["options"].get("seed_ids", []) if job else []


class OntologyJobRunner:
    """Single-worker background runner for ontology jobs.

    Builds, merges and reasoning runs share the one worker, so they never write to the
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
        if job.get("job_type") == "merge":
            await self._run_merge(job, log, set_progress)
            return
        config_dir = tempfile.mkdtemp(prefix=f"cogmaps_olaf_{job_id}_")
        try:
            self._write_config(config_dir, job)
            export_cb = _ttl_writer(collection, job["ontology_id"])

            # ── Domain guidance (optional, chosen per job) ──
            blueprint = None
            domain_mode = job_domain_mode(job)
            if domain_mode == "manual":
                guidelines = load_manual_guidelines(collection)
                if guidelines:
                    blueprint = DomainBlueprint(inferred_domain=collection, guidelines=guidelines)
                    log("Domain guidance: using the hand-written guidelines.")
                else:
                    log("Domain guidance: no hand-written guidelines saved — using the default prompt.")
            elif domain_mode == "llm":
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
                    detail = "hand-written guidelines" if blueprint.guidelines else f"{len(blueprint.pillars)} pillars"
                    log(f"Domain discovery: loaded cached blueprint -> '{blueprint.inferred_domain}' ({detail}).")
            else:
                log("No domain guidance for this build — using the default prompt.")

            qdrant = QdrantStore(self.qdrant_host, self.qdrant_port).client
            if migrated := migrate_shared_chunk_status(qdrant, collection):
                log(f"{migrated} chunk(s) processed by the collection's default ontology before chunk status "
                    "was tracked per ontology: now marked processed for it only.")
            ensure_ontology_index(job["ontology_id"], config_dir, qdrant, log)

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
                    seed_ids=job["options"].get("seed_ids", []),
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
                log(f"Backup of the ontology before any change: {save_backup(collection, job['ontology_id'], turtle)}")

            ensure_ontology_index(
                job["ontology_id"], config_dir, QdrantStore(self.qdrant_host, self.qdrant_port).client, log,
            )
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
                    export_cb=_ttl_writer(collection, job["ontology_id"]),
                    max_rounds=options.get("max_rounds", 3),
                    fix_orphans=options.get("fix_orphans", True),
                    dedup=options.get("dedup", True),
                    enrich_disjointness=options.get("enrich_disjointness", True),
                    infer=options.get("infer", True),
                    review_inferences=options.get("review_inferences", True),
                    seed_ids=build_seed_ids(self.store, collection, job["ontology_id"]),
                )
            self.store.set_result(job_id, summary)
            self.store.set_status(job_id, "done")
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, "failed")
            log(f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

    async def _run_merge(self, job: dict, log: Callable[[str], None], set_progress) -> None:
        """Merge the job's two source ontologies into its (new, empty) ontology."""
        job_id, collection, options = job["id"], job["collection"], job["options"]
        config_dir = tempfile.mkdtemp(prefix=f"cogmaps_olaf_{job_id}_")
        try:
            self._write_config(config_dir, job)

            def alignment_cb(tsv: str) -> None:
                path = alignment_path_for(collection, job["ontology_id"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tsv)

            log(f"Starting OLAF merge of {options['reference']!r} (reference) and {options['other']!r} "
                f"into ontology_id={job['ontology_id']!r}")
            async with olaf_session(config_dir) as session:
                summary = await run_merge(
                    session,
                    model=job["llm_model"],
                    api_key=scaleway_api_key(),
                    ontology_id=job["ontology_id"],
                    reference_id=options["reference"],
                    other_id=options["other"],
                    reference_collection=options.get("reference_collection"),
                    other_collection=options.get("other_collection"),
                    min_confidence=options.get("min_confidence", 0.7),
                    log=log,
                    set_progress=set_progress,
                    alignment_cb=alignment_cb,
                    export_cb=_ttl_writer(collection, job["ontology_id"]),
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
        write_config(
            config_dir,
            qdrant_url=f"http://{self.qdrant_host}:{self.qdrant_port}",
            qdrant_collection=job["collection"],
            ontology_id=job["ontology_id"],
            ontology_name=ontology_display_name(job["collection"], job["ontology_id"]),
        )


def _ttl_writer(collection: str, ontology_id: str) -> Callable[[str], None]:
    """Callback persisting an exported Turtle to the ontology's on-disk fallback copy."""
    def export_cb(turtle: str) -> None:
        out_path = ttl_path_for(collection, ontology_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(turtle)
    return export_cb
