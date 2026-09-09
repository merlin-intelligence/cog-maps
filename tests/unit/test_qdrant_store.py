"""Unit tests for cogmaps.qdrant.store — notably the document-sampling cap added
to keep the /analyze corpus/ page's memory usage bounded on a large collection,
against a fake QdrantClient (no real Qdrant server needed).
"""
from __future__ import annotations

import datetime

import cogmaps.qdrant.store as store_mod
from qdrant_client import models

from cogmaps.qdrant.store import QdrantStore, date_range_filter, make_point


# ── https=False must always be passed alongside api_key ──────────────────
#
# Regression test: qdrant-client defaults `https=True` as soon as `api_key`
# is set (see qdrant_remote.py: `self._https = https if https is not None
# else api_key is not None`). Our docker-compose.yml runs plain HTTP on
# localhost (the API key is defense-in-depth, not TLS) — omitting an explicit
# `https=False` alongside `api_key=...` makes every request attempt a TLS
# handshake against a plain-HTTP server and fail with
# "SSL: WRONG_VERSION_NUMBER" the moment QDRANT_API_KEY is set.

def test_init_passes_https_false_alongside_api_key(monkeypatch):
    captured = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(store_mod, "QdrantClient", _fake_client)
    monkeypatch.setattr(store_mod, "qdrant_api_key", lambda: "some-key")
    QdrantStore(host="localhost", port=6333)

    assert captured.get("api_key") == "some-key"
    assert captured.get("https") is False


def test_is_reachable_passes_https_false_alongside_api_key(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_collections(self):
            return []

    monkeypatch.setattr(store_mod, "QdrantClient", _FakeClient)
    monkeypatch.setattr(store_mod, "qdrant_api_key", lambda: "some-key")
    assert QdrantStore.is_reachable("localhost", 6333) is True

    assert captured.get("api_key") == "some-key"
    assert captured.get("https") is False


class _FakePoint:
    def __init__(self, payload, vector=None):
        self.payload = payload
        self.vector = vector


class FakeQdrantClient:
    """Emulates just the subset of QdrantClient.scroll used by QdrantStore."""

    def __init__(self, points):
        self._points = points

    def scroll(self, collection_name, limit, offset=None, with_payload=None,
               with_vectors=None, scroll_filter=None):
        pool = self._points
        if scroll_filter is not None:
            allowed = set(scroll_filter.must[0].match.any)
            pool = [p for p in pool if p.payload["filename"] in allowed]
        offset = offset or 0
        batch = pool[offset:offset + limit]
        next_offset = offset + limit if offset + limit < len(pool) else None
        return batch, next_offset


def _make_corpus(n_docs: int, chunks_per_doc: int = 2) -> list[_FakePoint]:
    points = []
    for i in range(n_docs):
        fname = f"doc_{i:05d}.pdf"
        for c in range(chunks_per_doc):
            points.append(_FakePoint(
                {"filename": fname, "chunk_number": c, "text": f"{fname} chunk {c}"},
                vector=[0.1, 0.2, 0.3],
            ))
    return points


def _store_with(points) -> QdrantStore:
    store = QdrantStore.__new__(QdrantStore)  # bypass __init__ (no real connection)
    store._client = FakeQdrantClient(points)
    return store


# ── all_documents: sampling cap ───────────────────────────────────────────

def test_all_documents_no_cap_loads_everything():
    store = _store_with(_make_corpus(50))
    docs, total = store.all_documents("col", max_documents=None)
    assert total == 50
    assert len(docs) == 50


def test_all_documents_caps_and_reports_true_total():
    store = _store_with(_make_corpus(500))
    docs, total = store.all_documents("col", max_documents=20)
    assert total == 500
    assert len(docs) == 20


def test_all_documents_sampling_is_deterministic_for_same_seed():
    store = _store_with(_make_corpus(500))
    docs_a, _ = store.all_documents("col", max_documents=20, sample_seed=7)
    docs_b, _ = store.all_documents("col", max_documents=20, sample_seed=7)
    assert set(docs_a) == set(docs_b)


def test_all_documents_sampling_differs_for_different_seed():
    store = _store_with(_make_corpus(500))
    docs_a, _ = store.all_documents("col", max_documents=20, sample_seed=1)
    docs_b, _ = store.all_documents("col", max_documents=20, sample_seed=2)
    assert set(docs_a) != set(docs_b)


def test_all_documents_cap_above_corpus_size_loads_everything():
    store = _store_with(_make_corpus(10))
    docs, total = store.all_documents("col", max_documents=1000)
    assert total == 10
    assert len(docs) == 10


def test_all_documents_reconstructs_text_in_chunk_order_and_means_vectors():
    points = [
        _FakePoint({"filename": "a.pdf", "chunk_number": 1, "text": "world"}, vector=[0.0, 1.0]),
        _FakePoint({"filename": "a.pdf", "chunk_number": 0, "text": "hello"}, vector=[1.0, 0.0]),
    ]
    store = _store_with(points)
    docs, total = store.all_documents("col", max_documents=None)
    assert total == 1
    assert docs["a.pdf"]["text"] == "hello\nworld"
    assert docs["a.pdf"]["vector"] is not None


def test_all_documents_vector_none_when_no_chunk_has_one():
    points = [_FakePoint({"filename": "a.pdf", "chunk_number": 0, "text": "x"}, vector=None)]
    store = _store_with(points)
    docs, _ = store.all_documents("col")
    assert docs["a.pdf"]["vector"] is None


# ── make_point / date_range_filter ────────────────────────────────────────

def test_make_point_has_uuid_id_and_expected_payload():
    p = make_point("f.pdf", 3, "some text", [0.1, 0.2], "2024-01-01T00:00:00")
    assert p.payload["filename"] == "f.pdf"
    assert p.payload["chunk_number"] == 3
    assert p.payload["text"] == "some text"
    assert p.payload["ingestion_date"] == "2024-01-01T00:00:00"
    assert isinstance(p.id, str) and len(p.id) == 36  # UUID4 string form


def test_date_range_filter_covers_exactly_one_calendar_day():
    flt = date_range_filter(datetime.date(2024, 6, 15))
    assert isinstance(flt, models.Filter)
    cond = flt.must[0]
    # qdrant_client parses the ISO strings we build into real datetimes.
    assert cond.range.gte == datetime.datetime(2024, 6, 15, 0, 0)
    assert cond.range.lt == datetime.datetime(2024, 6, 16, 0, 0)
