"""Daemon thread that processes ingestion jobs from an in-memory queue."""
from __future__ import annotations

import os
import queue
import shutil
import threading
import traceback

from cogmaps.jobs.store import JobStore
from cogmaps.pipelines.ingest import Ingester
from cogmaps.qdrant.store import QdrantStore
from cogmaps.ui.components import get_embedder

# Remote sources (Google Drive / SharePoint) are downloaded under this root
# before ingestion. Once a job over it succeeds, the downloaded copy is
# deleted — otherwise potentially sensitive company documents would sit in
# cleartext on the server disk indefinitely, long after they're safely in
# Qdrant. Never applied to admin-supplied local paths (those aren't ours to
# delete) — only to directories the app itself created under this root.
_DOWNLOADED_CORPUS_ROOT = os.path.realpath("downloaded_corpus")


class JobRunner:
    """Single-worker background runner.

    One daemon thread per (qdrant_host, qdrant_port) pair, shared across all
    browser sessions. Jobs are processed sequentially to avoid GPU contention.
    Survives browser disconnections and tab closures — only stops with the process.
    """

    def __init__(self, store: JobStore, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> None:
        self.store = store
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="eigenmind-ingestion"
        )
        self._thread.start()

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    # ── private ──────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            except Exception:  # noqa: BLE001
                self.store.set_status(job_id, "failed")
                self.store.append_log(job_id, f"Unexpected runner error:\n{traceback.format_exc()}")
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return

        self.store.set_status(job_id, "running")

        # Reuse the same process-wide embedder the Chat/Graph Explorer pages use
        # (Streamlit cache_resource) instead of loading a second copy of the model
        # for this job — avoids doubling GPU memory usage when ingestion and chat
        # run concurrently. EmbeddingModel serializes concurrent calls internally.
        ingester = Ingester(
            store=QdrantStore(self.qdrant_host, self.qdrant_port),
            device=job["device"],
            progress_callback=lambda cur, tot: self.store.set_progress(job_id, cur, tot),
            history_file=job["history_file"],
            embedder=get_embedder(job["device"]),
        )
        # Wire Ingester logs into the job store so they appear in the UI
        ingester._log = lambda msg: self.store.append_log(job_id, msg)

        succeeded = False
        try:
            ingester.run_chunknorris(job["directories_file"], job["collection"])
            if ingester.files_attempted > 0 and ingester.files_failed == ingester.files_attempted:
                self.store.set_status(job_id, "failed")
                self.store.append_log(
                    job_id,
                    f"All {ingester.files_attempted} file(s) failed to process — see errors above.",
                )
            else:
                succeeded = True
                self.store.set_status(job_id, "done")
        except Exception as e:  # noqa: BLE001
            self.store.set_status(job_id, "failed")
            self.store.append_log(job_id, f"Fatal: {e}\n{traceback.format_exc()}")
        finally:
            dirs_file = job["directories_file"]
            if succeeded:
                # Only on success — a failed job keeps its downloaded source
                # around so the user can retry without re-downloading.
                self._cleanup_downloaded_source(
                    dirs_file, log=lambda msg: self.store.append_log(job_id, msg)
                )
            try:
                if os.path.exists(dirs_file):
                    os.remove(dirs_file)
            except OSError:
                pass

    @staticmethod
    def _cleanup_downloaded_source(dirs_file: str, log) -> None:
        """Delete any ingested directory that lives under downloaded_corpus/."""
        try:
            with open(dirs_file, "r", encoding="utf-8", errors="ignore") as f:
                directories = [line.strip().strip("'\"") for line in f if line.strip()]
        except OSError:
            return
        for d in directories:
            real = os.path.realpath(d)
            if real == _DOWNLOADED_CORPUS_ROOT or real.startswith(_DOWNLOADED_CORPUS_ROOT + os.sep):
                shutil.rmtree(real, ignore_errors=True)
                log(f"Cleaned up downloaded source directory: {real}")
