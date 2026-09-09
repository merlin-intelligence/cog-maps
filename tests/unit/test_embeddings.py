"""Unit tests for cogmaps.core.embeddings: the release-after-use guard and the
lock that serializes concurrent inference (chat vs. a background ingestion job
sharing the same cached model instance).

A fake SentenceTransformer is substituted so these stay fast and independent of
downloading the real (large) embedding model / needing a GPU.
"""
from __future__ import annotations

import threading
import time

import pytest

from cogmaps.core import embeddings as embeddings_mod
from cogmaps.core.embeddings import EmbeddingModel


class _FakeSentenceTransformer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 768

    def encode(self, text):
        if isinstance(text, str):
            return [0.0] * 768
        return [[0.0] * 768 for _ in text]


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "SentenceTransformer", _FakeSentenceTransformer)


def test_release_then_use_raises_clear_runtime_error():
    model = EmbeddingModel(device="cpu")
    model.release()
    with pytest.raises(RuntimeError, match="released"):
        model.encode("hello")


def test_release_is_idempotent():
    model = EmbeddingModel(device="cpu")
    model.release()
    model.release()  # must not raise


def test_context_manager_releases_on_exit():
    with EmbeddingModel(device="cpu") as model:
        model.encode("hello")
    with pytest.raises(RuntimeError):
        model.encode("hello")


def test_dim_and_raw_also_guarded_after_release():
    model = EmbeddingModel(device="cpu")
    model.release()
    with pytest.raises(RuntimeError):
        _ = model.dim
    with pytest.raises(RuntimeError):
        _ = model.raw


def test_concurrent_encode_calls_are_serialized():
    """Two threads (standing in for a chat request and a background ingestion
    job) must never run inference on the shared instance at the same time."""
    model = EmbeddingModel(device="cpu")
    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    real_encode = model._model.encode

    def _tracked_encode(text):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.02)
        with state_lock:
            state["active"] -= 1
        return real_encode(text)

    model._model.encode = _tracked_encode

    def _worker(method, text):
        for _ in range(5):
            getattr(model, method)(text)

    t1 = threading.Thread(target=_worker, args=("encode_passage", "chunk text"))
    t2 = threading.Thread(target=_worker, args=("encode_query", "a question"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert state["max_active"] == 1
