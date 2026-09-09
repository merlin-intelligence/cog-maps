"""Unit tests for cogmaps.jobs.runner.JobRunner: job status transitions (done /
failed / "all files failed"), and that the shared cached embedder is fetched
and injected into the Ingester instead of letting it load its own.

Ingester, QdrantStore and get_embedder are faked out — this tests JobRunner's
own orchestration logic, not the real ingestion/embedding pipeline (covered
separately in test_ingest_pipeline.py and test_embeddings.py).
"""
from __future__ import annotations

import cogmaps.jobs.runner as runner_mod
from cogmaps.jobs.runner import JobRunner
from cogmaps.jobs.store import JobStore


class _FakeIngester:
    """Records how it was constructed and lets tests script run_chunknorris's outcome."""

    next_files_attempted = 0
    next_files_failed = 0
    next_exception: Exception | None = None
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _FakeIngester.last_kwargs = kwargs
        self._log = lambda msg: None
        self.files_attempted = 0
        self.files_failed = 0

    def run_chunknorris(self, directories_file, collection):
        if _FakeIngester.next_exception is not None:
            raise _FakeIngester.next_exception
        self.files_attempted = _FakeIngester.next_files_attempted
        self.files_failed = _FakeIngester.next_files_failed
        return []


def _reset_fake_ingester():
    _FakeIngester.next_files_attempted = 0
    _FakeIngester.next_files_failed = 0
    _FakeIngester.next_exception = None
    _FakeIngester.last_kwargs = None


def _runner_with(store) -> JobRunner:
    runner = JobRunner.__new__(JobRunner)  # skip __init__: no real thread/connection
    runner.store = store
    runner.qdrant_host = "localhost"
    runner.qdrant_port = 6333
    return runner


def _patch_runner_deps(monkeypatch, embedder_sentinel="fake-embedder"):
    _reset_fake_ingester()
    monkeypatch.setattr(runner_mod, "Ingester", _FakeIngester)
    monkeypatch.setattr(runner_mod, "QdrantStore", lambda host, port: f"fake-store-{host}-{port}")
    monkeypatch.setattr(runner_mod, "get_embedder", lambda device: f"{embedder_sentinel}-{device}")


def _submit_and_run(store, runner, device="cpu"):
    job_id = "job-1"
    store.create_job(
        job_id=job_id, collection="col", directories_file="/tmp/dirs.txt",
        device=device, history_file=None,
    )
    runner._run(job_id)
    return job_id


def test_run_missing_job_is_a_no_op(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    runner = _runner_with(store)
    runner._run("no-such-job")  # must not raise


def test_successful_ingestion_marks_job_done(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 3
    _FakeIngester.next_files_failed = 0

    runner = _runner_with(store)
    job_id = _submit_and_run(store, runner)

    assert store.get_job(job_id)["status"] == "done"


def test_zero_files_to_process_is_still_done_not_failed(monkeypatch, tmp_path):
    """'No new files to index' (attempted=0) is a legitimate success, distinct
    from 'every attempted file errored out'."""
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 0
    _FakeIngester.next_files_failed = 0

    runner = _runner_with(store)
    job_id = _submit_and_run(store, runner)

    assert store.get_job(job_id)["status"] == "done"


def test_all_files_failed_marks_job_failed(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 4
    _FakeIngester.next_files_failed = 4

    runner = _runner_with(store)
    job_id = _submit_and_run(store, runner)

    job = store.get_job(job_id)
    assert job["status"] == "failed"
    logs = [msg for _, msg in store.get_logs(job_id)]
    assert any("All 4 file(s) failed" in m for m in logs)


def test_partial_failure_still_marks_job_done(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 4
    _FakeIngester.next_files_failed = 2  # some succeeded

    runner = _runner_with(store)
    job_id = _submit_and_run(store, runner)

    assert store.get_job(job_id)["status"] == "done"


def test_exception_during_ingestion_marks_job_failed_with_log(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_exception = RuntimeError("qdrant unreachable")

    runner = _runner_with(store)
    job_id = _submit_and_run(store, runner)

    job = store.get_job(job_id)
    assert job["status"] == "failed"
    logs = [msg for _, msg in store.get_logs(job_id)]
    assert any("Fatal: qdrant unreachable" in m for m in logs)


def test_directories_file_removed_after_run(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    dirs_file = tmp_path / "dirs.txt"
    dirs_file.write_text("/some/dir\n")

    runner = _runner_with(store)
    job_id = "job-1"
    store.create_job(job_id=job_id, collection="col", directories_file=str(dirs_file),
                      device="cpu", history_file=None)
    runner._run(job_id)

    assert not dirs_file.exists()


def test_successful_job_deletes_downloaded_source_directory(monkeypatch, tmp_path):
    """A remote (GDrive/SharePoint) download must not linger in cleartext on
    disk after it has been safely ingested into Qdrant."""
    downloaded_root = tmp_path / "downloaded_corpus"
    source_dir = downloaded_root / "alice" / "my_collection"
    source_dir.mkdir(parents=True)
    (source_dir / "doc.pdf").write_text("secret content")
    monkeypatch.setattr(runner_mod, "_DOWNLOADED_CORPUS_ROOT", str(downloaded_root.resolve()))

    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 1
    _FakeIngester.next_files_failed = 0

    dirs_file = tmp_path / "dirs.txt"
    dirs_file.write_text(str(source_dir) + "\n")
    runner = _runner_with(store)
    job_id = "job-1"
    store.create_job(job_id=job_id, collection="col", directories_file=str(dirs_file),
                      device="cpu", history_file=None)
    runner._run(job_id)

    assert store.get_job(job_id)["status"] == "done"
    assert not source_dir.exists()
    logs = [msg for _, msg in store.get_logs(job_id)]
    assert any("Cleaned up downloaded source directory" in m for m in logs)


def test_failed_job_keeps_downloaded_source_directory_for_retry(monkeypatch, tmp_path):
    downloaded_root = tmp_path / "downloaded_corpus"
    source_dir = downloaded_root / "alice" / "my_collection"
    source_dir.mkdir(parents=True)
    (source_dir / "doc.pdf").write_text("secret content")
    monkeypatch.setattr(runner_mod, "_DOWNLOADED_CORPUS_ROOT", str(downloaded_root.resolve()))

    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 1
    _FakeIngester.next_files_failed = 1  # all failed

    dirs_file = tmp_path / "dirs.txt"
    dirs_file.write_text(str(source_dir) + "\n")
    runner = _runner_with(store)
    job_id = "job-1"
    store.create_job(job_id=job_id, collection="col", directories_file=str(dirs_file),
                      device="cpu", history_file=None)
    runner._run(job_id)

    assert store.get_job(job_id)["status"] == "failed"
    assert source_dir.exists()  # kept around so the user can retry


def test_admin_local_path_outside_downloaded_corpus_is_never_deleted(monkeypatch, tmp_path):
    """An admin-supplied arbitrary server path (Folder path / Paths file modes)
    must never be touched — the app doesn't own that directory."""
    downloaded_root = tmp_path / "downloaded_corpus"
    monkeypatch.setattr(runner_mod, "_DOWNLOADED_CORPUS_ROOT", str(downloaded_root.resolve()))

    admin_dir = tmp_path / "some" / "admin_owned_path"
    admin_dir.mkdir(parents=True)
    (admin_dir / "report.pdf").write_text("not ours to delete")

    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch)
    _FakeIngester.next_files_attempted = 1
    _FakeIngester.next_files_failed = 0

    dirs_file = tmp_path / "dirs.txt"
    dirs_file.write_text(str(admin_dir) + "\n")
    runner = _runner_with(store)
    job_id = "job-1"
    store.create_job(job_id=job_id, collection="col", directories_file=str(dirs_file),
                      device="cpu", history_file=None)
    runner._run(job_id)

    assert store.get_job(job_id)["status"] == "done"
    assert admin_dir.exists()
    assert (admin_dir / "report.pdf").exists()


def test_shared_cached_embedder_is_fetched_and_injected(monkeypatch, tmp_path):
    """The GPU-contention fix: the job must reuse get_embedder(device) rather
    than letting Ingester load its own separate copy of the model."""
    store = JobStore(str(tmp_path / "jobs.db"))
    _patch_runner_deps(monkeypatch, embedder_sentinel="shared-embedder")

    runner = _runner_with(store)
    _submit_and_run(store, runner, device="cuda")

    assert _FakeIngester.last_kwargs["embedder"] == "shared-embedder-cuda"
    assert _FakeIngester.last_kwargs["device"] == "cuda"
