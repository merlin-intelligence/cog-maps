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

import tempfile
from pathlib import Path

import numpy as np
from pyvis.network import Network

from cogmaps.config import NODE_BASE_SIZE, NODE_SIZE_MULTIPLIER
from cogmaps.graph.explorer import (
    _freeze_physics_after_stabilization,
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

