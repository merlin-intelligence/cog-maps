"""Unit tests for cogmaps.ontology.store.OntologyJobStore: job lifecycle, logs, and
the crash-recovery behavior (interrupted jobs marked 'failed' on the next startup).

Mirrors tests/unit/test_job_store.py for cogmaps.jobs.store.JobStore.
"""
from __future__ import annotations

from cogmaps.ontology.store import OntologyJobStore


def _store(tmp_path) -> OntologyJobStore:
    return OntologyJobStore(str(tmp_path / "ontology_jobs.db"))


def _create(store: OntologyJobStore, job_id: str, collection: str = "col") -> None:
    store.create_job(
        job_id=job_id, collection=collection, ontology_id="main",
        doc_filenames=["a.pdf", "b.pdf"], llm_provider="scaleway",
        llm_model="moonshotai/Kimi-K2.5-fast",
    )


def test_create_and_get_job_round_trips(tmp_path):
    store = _store(tmp_path)
    _create(store, "j1")
    job = store.get_job("j1")
    assert job is not None
    assert job["status"] == "pending"
    assert job["collection"] == "col"
    assert job["ontology_id"] == "main"
    assert job["doc_filenames"] == ["a.pdf", "b.pdf"]
    assert job["llm_model"] == "moonshotai/Kimi-K2.5-fast"
    assert job["progress_current"] == 0
    assert job["progress_total"] == 0


def test_get_job_returns_none_for_unknown_id(tmp_path):
    store = _store(tmp_path)
    assert store.get_job("does-not-exist") is None


def test_set_status_updates_job(tmp_path):
    store = _store(tmp_path)
    _create(store, "j1")
    store.set_status("j1", "running")
    assert store.get_job("j1")["status"] == "running"
    store.set_status("j1", "done")
    assert store.get_job("j1")["status"] == "done"


def test_set_progress_updates_current_and_total(tmp_path):
    store = _store(tmp_path)
    _create(store, "j1")
    store.set_progress("j1", 3, 10)
    job = store.get_job("j1")
    assert job["progress_current"] == 3
    assert job["progress_total"] == 10


def test_append_log_and_get_logs_ordering_and_cursor(tmp_path):
    store = _store(tmp_path)
    _create(store, "j1")
    store.append_log("j1", "first")
    store.append_log("j1", "second")
    store.append_log("j1", "third")

    all_logs = store.get_logs("j1")
    assert [msg for _, msg in all_logs] == ["first", "second", "third"]

    first_rowid = all_logs[0][0]
    remaining = store.get_logs("j1", after_rowid=first_rowid)
    assert [msg for _, msg in remaining] == ["second", "third"]


def test_logs_are_scoped_to_their_job(tmp_path):
    store = _store(tmp_path)
    _create(store, "j1")
    _create(store, "j2", collection="col2")
    store.append_log("j1", "for j1")
    store.append_log("j2", "for j2")
    assert [msg for _, msg in store.get_logs("j1")] == ["for j1"]
    assert [msg for _, msg in store.get_logs("j2")] == ["for j2"]


def test_latest_job_for_collection_picks_most_recent(tmp_path):
    store = _store(tmp_path)
    _create(store, "older")
    _create(store, "newer")
    _create(store, "other-col", collection="other")

    latest = store.latest_job_for("col")
    assert latest["id"] == "newer"


def test_latest_job_for_unknown_collection_is_none(tmp_path):
    store = _store(tmp_path)
    assert store.latest_job_for("nonexistent") is None


def test_startup_marks_interrupted_jobs_failed(tmp_path):
    """A job left 'running' or 'pending' when the process died must be marked
    'failed' the next time an OntologyJobStore opens that DB — otherwise it
    would hang as 'running' forever with no worker ever processing it."""
    db_path = tmp_path / "ontology_jobs.db"
    store1 = OntologyJobStore(str(db_path))
    _create(store1, "was-running")
    store1.set_status("was-running", "running")
    _create(store1, "was-pending")
    _create(store1, "was-done")
    store1.set_status("was-done", "done")

    # Simulate a process restart: open a fresh OntologyJobStore against the same DB file.
    store2 = OntologyJobStore(str(db_path))
    assert store2.get_job("was-running")["status"] == "failed"
    assert store2.get_job("was-pending")["status"] == "failed"
    assert store2.get_job("was-done")["status"] == "done"  # untouched


def test_use_domain_discovery_defaults_to_false(tmp_path):
    store = _store(tmp_path)
    _create(store, "j1")
    assert store.get_job("j1")["use_domain_discovery"] is False


def test_use_domain_discovery_round_trips(tmp_path):
    store = _store(tmp_path)
    store.create_job(
        job_id="j1", collection="col", ontology_id="main",
        doc_filenames=["a.pdf"], llm_provider="scaleway", llm_model="m",
        use_domain_discovery=True,
    )
    assert store.get_job("j1")["use_domain_discovery"] is True
    assert store.latest_job_for("col")["use_domain_discovery"] is True


def test_existing_db_without_discovery_column_is_migrated(tmp_path):
    import sqlite3

    db = tmp_path / "ontology_jobs.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE ontology_jobs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
            collection TEXT NOT NULL, ontology_id TEXT NOT NULL,
            doc_filenames_json TEXT NOT NULL, llm_provider TEXT NOT NULL,
            llm_model TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            progress_current INTEGER NOT NULL DEFAULT 0, progress_total INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute(
        "INSERT INTO ontology_jobs VALUES ('old', 'done', 'col', 'main', '[]', 'scaleway', 'm', 't', 't', 0, 0)"
    )
    conn.commit()
    conn.close()

    store = OntologyJobStore(str(db))
    assert store.get_job("old")["use_domain_discovery"] is False
    _create(store, "new")
    assert store.get_job("new")["use_domain_discovery"] is False
