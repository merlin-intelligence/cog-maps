"""Unit tests for cogmaps.core.corpus_analysis: stats, and the two analyses
(topic_model, similar_pairs) that must gracefully exclude documents missing
an embedding rather than failing on the whole corpus.
"""
from __future__ import annotations

import numpy as np

from cogmaps.core.corpus_analysis import (
    CorpusDocument,
    length_stats,
    similar_pairs,
    to_dataframe,
    topic_model,
    type_counts,
)


def _doc(title: str, content: str = "some text", ext: str = ".pdf", vector=None) -> CorpusDocument:
    return CorpusDocument.from_filename(f"{title}{ext}", content, vector)


# ── to_dataframe / type_counts / length_stats ────────────────────────────

def test_to_dataframe_and_type_counts():
    docs = [_doc("a", "hello", ".pdf"), _doc("b", "hi there", ".docx")]
    df = to_dataframe(docs)
    assert list(df["title"]) == ["a", "b"]
    assert list(df["type"]) == ["pdf", "docx"]
    counts = type_counts(df)
    assert counts.iloc[0].to_dict() == {"Type": "Total", "Count": 2}


def test_length_stats_on_empty_dataframe():
    df = to_dataframe([])
    stats = length_stats(df)
    assert stats == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


# ── similar_pairs: vectorized, excludes vectorless docs, threshold edge case ──

def _unit_vec(*coords) -> np.ndarray:
    v = np.array(coords, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_similar_pairs_finds_identical_and_excludes_orthogonal():
    docs = [
        _doc("a", vector=_unit_vec(1.0, 0.0)),
        _doc("b", vector=_unit_vec(1.0, 0.0)),
        _doc("c", vector=_unit_vec(0.0, 1.0)),
    ]
    pairs = similar_pairs(docs, threshold=0.99)
    assert len(pairs) == 1
    assert {pairs[0].a, pairs[0].b} == {"a", "b"}
    assert pairs[0].score > 0.99


def test_similar_pairs_excludes_docs_without_a_vector_instead_of_disabling_everything():
    docs = [
        _doc("a", vector=_unit_vec(1.0, 0.0)),
        _doc("b", vector=None),  # no stored embedding
        _doc("c", vector=_unit_vec(1.0, 0.0)),
    ]
    pairs = similar_pairs(docs, threshold=0.99)
    assert len(pairs) == 1
    assert {pairs[0].a, pairs[0].b} == {"a", "c"}


def test_similar_pairs_threshold_zero_or_negative_has_no_false_positive_self_pairs():
    docs = [
        _doc("a", vector=_unit_vec(1.0, 0.0)),
        _doc("b", vector=_unit_vec(0.0, 1.0)),
    ]
    pairs = similar_pairs(docs, threshold=-1.0)
    # Exactly the one i<j pair, never a self-pair or a duplicate from the
    # lower triangle.
    assert len(pairs) == 1
    assert {pairs[0].a, pairs[0].b} == {"a", "b"}


def test_similar_pairs_returns_empty_with_fewer_than_two_usable_docs():
    docs = [_doc("a", vector=_unit_vec(1.0, 0.0)), _doc("b", vector=None)]
    assert similar_pairs(docs) == []


# ── topic_model: excludes vectorless docs, keeps df/labels aligned ──────

def test_topic_model_excludes_vectorless_docs_and_keeps_df_aligned():
    docs = [
        _doc("a", content="cats and dogs", vector=_unit_vec(1.0, 0.0, 0.0)),
        _doc("b", content="no embedding here", vector=None),
        _doc("c", content="cats and dogs again", vector=_unit_vec(0.9, 0.1, 0.0)),
        _doc("d", content="rockets and space", vector=_unit_vec(0.0, 0.0, 1.0)),
    ]
    df = to_dataframe(docs)
    model = topic_model(docs, df, stopwords=set(), k_min=2, k_max=3)
    assert model is not None
    # Only the 3 docs with vectors are clustered.
    assert len(model.labels) == 3
    assert len(model.cluster_label) == model.best_k


def test_topic_model_returns_none_below_k_min_usable_docs():
    docs = [_doc("a", vector=_unit_vec(1.0, 0.0)), _doc("b", vector=None)]
    df = to_dataframe(docs)
    assert topic_model(docs, df, stopwords=set(), k_min=2) is None


def test_topic_model_returns_none_with_exactly_k_min_usable_docs_no_crash():
    """With exactly k_min (2) usable docs, n_clusters would equal n_samples,
    which is outside sklearn's silhouette_score domain (2 <= n_labels <=
    n_samples - 1); topic_model must return None instead of raising."""
    docs = [
        _doc("a", vector=_unit_vec(1.0, 0.0)),
        _doc("b", vector=_unit_vec(0.0, 1.0)),
        _doc("c", vector=None),
    ]
    df = to_dataframe(docs)
    assert topic_model(docs, df, stopwords=set(), k_min=2) is None


def test_topic_model_succeeds_with_exactly_k_min_plus_one_usable_docs():
    docs = [
        _doc("a", content="cats and dogs", vector=_unit_vec(1.0, 0.0)),
        _doc("b", content="cats again", vector=_unit_vec(0.9, 0.1)),
        _doc("c", content="rockets and space", vector=_unit_vec(0.0, 1.0)),
    ]
    df = to_dataframe(docs)
    model = topic_model(docs, df, stopwords=set(), k_min=2)
    assert model is not None
    assert len(model.labels) == 3
