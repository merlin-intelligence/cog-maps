"""Unit tests for cogmaps.graph.explorer._scale_node_sizes and
_freeze_physics_after_stabilization.

_scale_node_sizes: on a small or textually homogeneous corpus, the
similarity graph is near-complete, so every node's degree is uniformly
high; sizes must stay clamped to a legible range regardless
("les points sont beaucoup trop gros, on ne voit rien").

_freeze_physics_after_stabilization: vis-network keeps its force simulation
running forever after the initial stabilization, so nodes visibly jitter
continuously ("les points vibrent"). pyvis's own stabilization handler only
hides the loading bar — it never disables physics.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from eigenmind import SimilarityGraph
from pyvis.network import Network

from cogmaps.config import NODE_BASE_SIZE, NODE_SIZE_MULTIPLIER
from cogmaps.graph.explorer import (
    _freeze_physics_after_stabilization,
    _plot_laplacian_spectrum,
    _query_utilities,
    _scale_node_sizes,
    select_adaptive_threshold,
)

_MAX_SIZE = NODE_BASE_SIZE + NODE_SIZE_MULTIPLIER * 10


def test_sizes_always_bounded_regardless_of_absolute_degree():
    # A small, textually homogeneous corpus: every chunk connects to nearly
    # every other one, so degree is uniformly high — exactly the reported bug.
    degrees = np.array([18, 19, 17, 18, 19, 18, 17, 18])
    sizes = _scale_node_sizes(degrees)
    assert max(sizes) <= _MAX_SIZE
    assert min(sizes) >= NODE_BASE_SIZE


def test_zero_degree_nodes_get_the_base_size():
    sizes = _scale_node_sizes(np.array([0, 0, 0]))
    assert sizes == [NODE_BASE_SIZE, NODE_BASE_SIZE, NODE_BASE_SIZE]


def test_empty_graph_returns_empty_list():
    assert _scale_node_sizes(np.array([])) == []


def test_hub_node_is_still_the_largest_among_mostly_isolated_nodes():
    degrees = np.array([0, 1, 1, 50, 2])
    sizes = _scale_node_sizes(degrees)
    assert sizes[3] == max(sizes)
    assert max(sizes) <= _MAX_SIZE
    assert sizes[0] == NODE_BASE_SIZE  # isolated node stays at the floor


def test_sizes_are_monotonic_in_degree():
    degrees = np.array([0, 2, 5, 10, 20])
    sizes = _scale_node_sizes(degrees)
    assert sizes == sorted(sizes)


# ── _freeze_physics_after_stabilization ──────────────────────────────────

def test_freeze_physics_injects_a_setoptions_call_before_body_close(tmp_path):
    net = Network(height="900px", width="100%", notebook=False, directed=False)
    net.add_node(1, label="a")
    net.add_node(2, label="b")
    net.add_edge(1, 2)
    path = str(tmp_path / "graph.html")
    net.save_graph(path)

    _freeze_physics_after_stabilization(path)

    html_content = Path(path).read_text(encoding="utf-8")
    assert "network.setOptions({ physics: false })" in html_content
    # The freeze script must run before </body>, and reference the same
    # `network` variable pyvis's own script assigns (declared with `var` at
    # the top level, so it's reachable from an appended <script> tag).
    assert html_content.index("network.setOptions({ physics: false })") < html_content.rindex("</body>")
    assert 'typeof network !== "undefined"' in html_content


# ── select_adaptive_threshold ──────────────────────────────────────────

def test_select_adaptive_threshold_small_matrix():
    # <= 2 nodes defaults to 0.65 safely
    assert select_adaptive_threshold(np.ones((2, 2))) == 0.65
    assert select_adaptive_threshold(np.ones((1, 1))) == 0.65


def test_select_adaptive_threshold_separates_dense_components():
    # 30 nodes: 15 in cluster A (cosines ~0.95), 15 in cluster B (cosines ~0.95),
    # between A and B cosines are 0.90.
    n = 30
    C = np.full((n, n), 0.90)
    C[:15, :15] = 0.95
    C[15:, 15:] = 0.95
    np.fill_diagonal(C, 1.0)

    tau = select_adaptive_threshold(C, min_components=2, max_dominant_ratio=0.85)
    # Threshold must separate cluster A and cluster B (tau >= 0.90)
    assert tau >= 0.90
    W = np.where(C > tau, C, 0.0)
    np.fill_diagonal(W, 0.0)
    from scipy.sparse.csgraph import connected_components
    n_comps, _ = connected_components(W, directed=False)
    assert n_comps >= 2


def test_select_adaptive_threshold_preserves_base_if_already_separated():
    # 20 nodes with low inter-cluster similarity already separated at q=0.88
    n = 20
    C = np.full((n, n), 0.50)
    C[:10, :10] = 0.85
    C[10:, 10:] = 0.85
    np.fill_diagonal(C, 1.0)

    tau = select_adaptive_threshold(C, min_components=2)
    assert tau >= 0.65


def test_select_adaptive_threshold_ignores_isolated_chunks():
    # Nodes 0-9 form one cluster (cosine 0.92) holding two tighter triangles
    # (0.97); nodes 10-11 are outliers. At q=0.1 (tau=0.65) the graph is the
    # cluster plus two isolates: 3 components when isolates count, which was
    # accepted although only one of them can be bridged. Only q=0.5 (tau=0.92)
    # yields two real components.
    n = 12
    C = np.full((n, n), 0.10)
    C[:10, :10] = 0.92
    C[:3, :3] = C[3:6, 3:6] = 0.97
    C[10, 11] = C[11, 10] = 0.60
    np.fill_diagonal(C, 1.0)

    tau = select_adaptive_threshold(C, min_components=2, candidates=(0.1, 0.5))
    assert tau == 0.92


# ── eigenmind 0.2 helpers ──────────────────────────────────────────────

def _graph(vectors):
    return SimilarityGraph([
        SimpleNamespace(id=i, vector=v, payload={"text": f"t{i}"})
        for i, v in enumerate(vectors)
    ])


def test_query_utilities_are_clipped_cosines_to_the_query():
    graph = _graph([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    u = _query_utilities(graph, [2.0, 0.0])
    np.testing.assert_allclose(u, [1.0, 0.0, 0.0])


def test_query_utilities_fall_back_to_unit_for_zero_query():
    assert _query_utilities(_graph([[1.0, 0.0], [0.0, 1.0]]), [0.0, 0.0]) is None


def test_plot_laplacian_spectrum_writes_png(tmp_path):
    graph = _graph(np.eye(4) + 0.5)
    path = _plot_laplacian_spectrum(graph, str(tmp_path / "q"))
    assert path == str(tmp_path / "q_eigenvalues.png")
    assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ── GraphExplorer.explore edge cases ───────────────────────────────────

def _explore(retrieved, tmp_path):
    from unittest.mock import MagicMock, patch

    from cogmaps.graph.explorer import GraphExplorer

    embedder = MagicMock()
    embedder.encode_query.return_value = np.ones(4)
    explorer = GraphExplorer(nlp=MagicMock(), store=MagicMock(), embedder=embedder)
    with patch("cogmaps.graph.explorer.explore_graph_with_initial_set",
               return_value=(retrieved, set())):
        return explorer.explore("c", "q", output_dir=str(tmp_path), render_artifacts=False)


def test_explore_handles_empty_retrieval(tmp_path, caplog):
    # graph.summary() raises on an empty graph; the INFO log must skip it.
    caplog.set_level("INFO", logger="cogmaps.graph.explorer")
    assert _explore([], tmp_path).ranked_chunks == []


def test_explore_survives_a_supported_diversity_failure(tmp_path):
    from unittest.mock import patch

    rng = np.random.default_rng(0)
    points = [SimpleNamespace(id=i, vector=rng.normal(size=4),
                              payload={"text": f"t{i}", "filename": "f"}) for i in range(8)]
    with patch.object(SimilarityGraph, "supported_diversity", side_effect=RuntimeError("LP failed")):
        ranked = _explore(points, tmp_path).ranked_chunks
    assert all("supported_diversity" not in c["methods"] for c in ranked)
