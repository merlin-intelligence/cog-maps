"""SQLite-backed persistence for background ontology-building jobs.

Mirrors :class:`cogmaps.jobs.store.JobStore` exactly (same SQLite/WAL shape,
same method surface) but kept as a separate table rather than overloading
the ingestion-specific ``directories_file``/``device`` columns there.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS ontology_jobs (
    id                TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'pending',
    collection        TEXT NOT NULL,
    ontology_id       TEXT NOT NULL,
    doc_filenames_json TEXT NOT NULL,
    llm_provider      TEXT NOT NULL,
    llm_model         TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    progress_current  INTEGER NOT NULL DEFAULT 0,
    progress_total    INTEGER NOT NULL DEFAULT 0,
    use_domain_discovery INTEGER NOT NULL DEFAULT 0,
    job_type          TEXT NOT NULL DEFAULT 'build',
    options_json      TEXT NOT NULL DEFAULT '{}',
    result_json       TEXT
);

CREATE TABLE IF NOT EXISTS ontology_job_logs (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT    NOT NULL,
    message TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ontology_job_llm_calls (
    rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    TEXT    NOT NULL,
    call_json TEXT    NOT NULL
);
"""


def _job_from_row(row: sqlite3.Row) -> dict:
    job = dict(row)
    job["doc_filenames"] = json.loads(job["doc_filenames_json"])
    job["use_domain_discovery"] = bool(job["use_domain_discovery"])
    job["options"] = json.loads(job["options_json"])
    job["result"] = json.loads(job["result_json"]) if job["result_json"] else None
    return job


class OntologyJobStore:
    """Thread-safe SQLite store for ontology-building job state and logs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Databases created before the domain-discovery option lack the column.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(ontology_jobs)")}
            if "use_domain_discovery" not in columns:
                conn.execute(
                    "ALTER TABLE ontology_jobs ADD COLUMN use_domain_discovery INTEGER NOT NULL DEFAULT 0"
                )
            # ... and those created before reasoning jobs, the job_type column.
            if "job_type" not in columns:
                conn.execute("ALTER TABLE ontology_jobs ADD COLUMN job_type TEXT NOT NULL DEFAULT 'build'")
            if "options_json" not in columns:
                conn.execute("ALTER TABLE ontology_jobs ADD COLUMN options_json TEXT NOT NULL DEFAULT '{}'")
            if "result_json" not in columns:
                conn.execute("ALTER TABLE ontology_jobs ADD COLUMN result_json TEXT")
            # On startup, any job that was 'running' or 'pending' was interrupted
            # by a server restart — mark them failed so they don't hang forever.
            conn.execute(
                "UPDATE ontology_jobs SET status='failed', updated_at=? WHERE status IN ('running', 'pending')",
                (datetime.now().isoformat(),),
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_job(
        self,
        job_id: str,
        collection: str,
        ontology_id: str,
        doc_filenames: list[str],
        llm_provider: str,
        llm_model: str,
        use_domain_discovery: bool = False,
        job_type: str = "build",
        options: dict | None = None,
    ) -> None:
        """``job_type`` is ``"build"`` (agent extracting the ontology from chunks) or
        ``"reasoning"`` (agent repairing the logical problems of an existing ontology);
        ``options`` holds the job type's own settings (e.g. a reasoning job's ``max_rounds``)."""
        now = datetime.now().isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO ontology_jobs
                   (id, status, collection, ontology_id, doc_filenames_json,
                    llm_provider, llm_model, created_at, updated_at, use_domain_discovery, job_type, options_json)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, collection, ontology_id, json.dumps(doc_filenames),
                 llm_provider, llm_model, now, now, int(use_domain_discovery), job_type,
                 json.dumps(options or {})),
            )

    def set_status(self, job_id: str, status: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE ontology_jobs SET status=?, updated_at=? WHERE id=?",
                (status, datetime.now().isoformat(), job_id),
            )

    def set_progress(self, job_id: str, current: int, total: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE ontology_jobs SET progress_current=?, progress_total=?, updated_at=? WHERE id=?",
                (current, total, datetime.now().isoformat(), job_id),
            )

    def set_result(self, job_id: str, result) -> None:
        """Store the job's JSON-serializable outcome (a reasoning job's problem counts per check)."""
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE ontology_jobs SET result_json=?, updated_at=? WHERE id=?",
                (json.dumps(result), datetime.now().isoformat(), job_id),
            )

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO ontology_job_logs (job_id, message) VALUES (?, ?)",
                (job_id, message),
            )

    def append_llm_call(self, job_id: str, call: dict) -> None:
        """Record one LLM request made by the job (an ``LLMCallImpact.to_dict()``)."""
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO ontology_job_llm_calls (job_id, call_json) VALUES (?, ?)",
                (job_id, json.dumps(call)),
            )

    def get_llm_calls(self, job_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT call_json FROM ontology_job_llm_calls WHERE job_id=? ORDER BY rowid",
                (job_id,),
            ).fetchall()
            return [json.loads(r["call_json"]) for r in rows]

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM ontology_jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            return _job_from_row(row)

    def get_logs(self, job_id: str, after_rowid: int = 0) -> list[tuple[int, str]]:
        """Return new log entries since after_rowid, as (rowid, message) pairs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT rowid, message FROM ontology_job_logs WHERE job_id=? AND rowid>? ORDER BY rowid",
                (job_id, after_rowid),
            ).fetchall()
            return [(r["rowid"], r["message"]) for r in rows]

    def latest_job_for(self, collection: str, job_type: str | None = None) -> dict | None:
        """The collection's most recent job — of ``job_type`` only, when given."""
        query = "SELECT * FROM ontology_jobs WHERE collection=?"
        params: tuple = (collection,)
        if job_type:
            query += " AND job_type=?"
            params += (job_type,)
        with self._conn() as conn:
            row = conn.execute(f"{query} ORDER BY created_at DESC LIMIT 1", params).fetchone()
            if not row:
                return None
            return _job_from_row(row)
