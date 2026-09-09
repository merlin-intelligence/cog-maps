"""Unit tests for cogmaps.graph.explorer._scale_node_sizes and
_freeze_physics_after_stabilization.

Regression 1: on a small or textually homogeneous corpus, the similarity
graph is near-complete, so every node's degree — and with the old unbounded
formula, its rendered size — was uniformly huge, making the graph
unreadable ("les points sont beaucoup trop gros, on ne voit rien").

Regression 2: vis-network keeps its force simulation running forever after
the initial stabilization, so nodes visibly jitter continuously
("les points vibrent"). pyvis's own stabilization handler only hides the
loading bar — it never disables physics.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pyvis.network import Network

from cogmaps.config import NODE_BASE_SIZE, NODE_SIZE_MULTIPLIER
from cogmaps.graph.explorer import _freeze_physics_after_stabilization, _scale_node_sizes

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
