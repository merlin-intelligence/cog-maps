"""Unit tests for cogmaps.pipelines.ingest: the single-pass directory walk, the
per-file failure tracking that drives the job-status "all files failed" check,
and that an injected (shared, cached) embedder is never released by the job.
"""
from __future__ import annotations

import os

from cogmaps.pipelines import ingest as ingest_mod
from cogmaps.pipelines.ingest import Ingester, _read_directories_file, _walk_and_classify

# ── _read_directories_file ────────────────────────────────────────────────

def test_read_directories_file_strips_quotes_and_blank_lines(tmp_path):
    f = tmp_path / "dirs.txt"
    f.write_text('/some/path\n"/quoted/path"\n\n\'/single/quoted\'\n')
    assert _read_directories_file(str(f), log=lambda _m: None) == [
        "/some/path", "/quoted/path", "/single/quoted",
    ]


def test_read_directories_file_missing_file_logs_and_returns_empty():
    logs = []
    result = _read_directories_file("/does/not/exist.txt", log=logs.append)
    assert result == []
    assert any("not found" in m for m in logs)


# ── _walk_and_classify: single-pass supported/unsupported/skip/ignore ────

def test_walk_and_classify_single_pass_handles_all_cases(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.json").write_text("x")          # unsupported, reported
    (sub / "c.docx").write_text("x")                # supported, in a subdir
    (tmp_path / "d.png").write_text("x")             # neither list -> silently ignored
    (tmp_path / "~$e.pdf").write_text("x")           # lock/temp file -> ignored
    (tmp_path / "already.pdf").write_text("x")       # already in Qdrant -> skipped

    logs = []
    files = _walk_and_classify(
        [str(tmp_path)],
        supported_extensions=(".pdf", ".docx"),
        unsupported_extensions=(".json",),
        skip={"already.pdf"},
        log=logs.append,
    )

    names = sorted(os.path.basename(f) for f in files)
    assert names == ["a.pdf", "c.docx"]
    assert any("b.json" in m and "not supported" in m for m in logs)
    assert any("Skipped 1 files already present" in m for m in logs)


def test_walk_and_classify_warns_on_invalid_directory():
    logs = []
    files = _walk_and_classify(
        ["/no/such/directory"], supported_extensions=(".pdf",),
        unsupported_extensions=(".json",), skip=set(), log=logs.append,
    )
    assert files == []
    assert any("not a valid directory" in m for m in logs)


# ── Ingester._chunknorris_points: per-file failure tracking ──────────────

class _FakeEmbedder:
    def encode_passage(self, text):
        class _Vec:
            def tolist(self_inner):
                return [0.0]
        return _Vec()


def test_chunknorris_points_tracks_total_failure(monkeypatch, tmp_path):
    """When every file errors out, files_failed must equal files_attempted —
    this is what JobRunner uses to mark the job 'failed' instead of 'done'."""
    def _always_fails(fp):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest_mod, "chunk_with_chunknorris", _always_fails)

    ing = Ingester(store=object(), device="cpu")
    files = [str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")]
    points = list(ing._chunknorris_points(files, _FakeEmbedder(), log=lambda _m: None))

    assert points == []
    assert ing.files_attempted == 2
    assert ing.files_failed == 2


def test_chunknorris_points_partial_failure_is_tracked_precisely(monkeypatch, tmp_path):
    def _fails_only_for_b(fp):
        if "b.pdf" in fp:
            raise RuntimeError("boom")
        chunk = type("Chunk", (), {"get_text": lambda self: "some text"})()
        return iter([chunk])

    monkeypatch.setattr(ingest_mod, "chunk_with_chunknorris", _fails_only_for_b)

    ing = Ingester(store=object(), device="cpu")
    files = [str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")]
    points = list(ing._chunknorris_points(files, _FakeEmbedder(), log=lambda _m: None))

    assert len(points) == 1
    assert ing.files_attempted == 2
    assert ing.files_failed == 1


# ── injected embedder is never released (the GPU-contention fix) ─────────

class _TrackedEmbedder:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_injected_embedder_scope_does_not_release_it():
    shared = _TrackedEmbedder()
    ing = Ingester(store=object(), device="cpu", embedder=shared)
    with ing._embedder_scope(log=lambda _m: None) as embedder:
        assert embedder is shared
    assert shared.released is False, (
        "an injected (shared, cache_resource-owned) embedder must never be "
        "released by the Ingester — only an embedder it created itself"
    )
