"""Graph-based RAG pipeline.

:class:`GraphExplorer` builds the artifacts shown by the Graph Explorer page:
HTML graphs, eigenvalue plot, ranked chunks, Excel export. Does NOT call the LLM.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from eigenmind import (
    SimilarityGraph,
    cosine_matrix,
    normalize_embeddings,
    quantile_threshold,
    threshold_graph,
)
from eigenmind.defaults import SIMILARITY_THRESHOLD
from matplotlib.figure import Figure
from pyvis.network import Network
from scipy.sparse.csgraph import connected_components

from cogmaps.config import (
    DIVERSITY_BUDGET,
    DIVERSITY_TIME_LIMIT,
    DOCUMENT_COLORS,
    GRAPH_BG_COLOR,
    GRAPH_EDGE_COLOR,
    GRAPH_METHODS_EDGE_COLOR,
    GRAPH_PROMPT_TEXT_COLOR,
    GRAPH_TEXT_COLOR,
    MAX_CHUNKS,
    METHOD_COLORS,
    NODE_BASE_SIZE,
    NODE_SIZE_MULTIPLIER,
    PROMPT_NODE_COLOR,
)
from cogmaps.core.embeddings import EmbeddingModel
from cogmaps.graph.exploration import explore_graph_with_initial_set
from cogmaps.qdrant.store import QdrantStore

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
#  Graph-Explorer pipeline (HTML + eigenvalues + Excel)
# ───────────────────────────────────────────────

@dataclass
class ExplorerArtifacts:
    """Files and data produced by :meth:`GraphExplorer.explore`."""
    graph_html: str
    eigenvalue_png: str
    summary_text: str
    methods_html: str
    excel_path: str
    ranked_chunks: list[dict] = field(default_factory=list)
    document_colors: dict[str, str] = field(default_factory=dict)


def _generate_filename_from_prompt(prompt: str, extension: str = ".html") -> str:
    short = " ".join(prompt.split()[:5])
    sanitized = re.sub(r"[^\w\s-]", "", short).strip().replace(" ", "_")
    return f"{sanitized}{extension}"


def _freeze_physics_after_stabilization(path: str) -> None:
    """Turn physics off once the initial layout settles, so nodes stop jittering.

    vis-network keeps the force simulation running forever after the
    `stabilization.iterations` configured in ``_PHYSICS`` — it never comes to
    rest on its own, so nodes visibly vibrate continuously. pyvis's own
    ``stabilizationIterationsDone`` handler only hides the loading bar; it
    never disables physics. ``network`` is declared with ``var`` at the top
    level of pyvis's generated script, so it's reachable from this appended
    one, once the page loads.
    """
    snippet = (
        '<script>\n'
        'if (typeof network !== "undefined") {\n'
        '  network.once("stabilizationIterationsDone", function () {\n'
        '    network.setOptions({ physics: false });\n'
        '  });\n'
        '}\n'
        '</script>\n'
    )
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", snippet + "</body>", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _scale_node_sizes(degrees: np.ndarray) -> list[int]:
    """Map node degrees to pyvis sizes, always within [NODE_BASE_SIZE, NODE_BASE_SIZE + NODE_SIZE_MULTIPLIER * 10].

    Sizing is relative to the *graph's own* max degree, log-compressed so a
    few hub nodes don't dominate, and clamped to a fixed, always-legible
    range regardless of the corpus — including on a small or textually
    homogeneous corpus, where the similarity graph is near-complete and
    every node's degree is otherwise uniformly huge.
    """
    if len(degrees) == 0:
        return []
    max_degree = float(degrees.max())
    if max_degree <= 0:
        return [NODE_BASE_SIZE] * len(degrees)
    span = NODE_SIZE_MULTIPLIER * 10
    log_max = np.log1p(max_degree)
    return [int(NODE_BASE_SIZE + (np.log1p(d) / log_max) * span) for d in degrees]


def _extract_themes(text: str, nlp, top_n: int = 5) -> list[str]:
    doc = nlp(text)
    keywords = [
        token.text.lower() for token in doc
        if token.pos_ in ("NOUN", "PROPN")
        and not token.is_stop and not token.is_punct
        and len(token.text) > 2
    ]
    return [item for item, _ in Counter(keywords).most_common(top_n)]


_PHYSICS = {
    "physics": {
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -10000, "centralGravity": 0.3,
            "springLength": 95, "springConstant": 0.04,
            "damping": 0.09, "avoidOverlap": 0.1,
        },
        "stabilization": {"iterations": 1000, "fit": True},
    },
    "edges": {"color": {"color": GRAPH_EDGE_COLOR, "highlight": GRAPH_EDGE_COLOR}},
    "configure": {"enabled": True, "filter": "physics", "showButton": True},
}

# The methods graph has far fewer nodes than the main graph (~15-30 vs. up to
# MAX_CHUNKS), so it needs a looser layout: a longer spring length and a much
# stronger avoidOverlap keep the small node set spread out instead of
# clumping together, which barnesHut's generic tuning above tends to do.
_METHODS_PHYSICS = {
    "physics": {
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -6000, "centralGravity": 0.2,
            "springLength": 180, "springConstant": 0.03,
            "damping": 0.12, "avoidOverlap": 0.8,
        },
        "stabilization": {"iterations": 1000, "fit": True},
    },
    "edges": {"color": {"color": GRAPH_METHODS_EDGE_COLOR, "highlight": GRAPH_METHODS_EDGE_COLOR}},
    "configure": {"enabled": True, "filter": "physics", "showButton": True},
}


def select_adaptive_threshold(
    C: np.ndarray,
    min_components: int = 3,
    max_dominant_ratio: float = 0.85,
    candidates: tuple[float, ...] = (0.88, 0.90, 0.92, 0.94, 0.95, 0.96),
) -> float:
    """Select an adaptive cosine threshold tau guaranteeing component separation.

    Evaluates candidate quantiles to ensure the similarity graph splits into at
    least ``min_components`` components of two or more chunks, without being
    completely dominated by a single giant component (> ``max_dominant_ratio``
    of nodes). Isolated chunks are not counted: ``semantic_bridges`` excludes
    them, so they would not give it anything to bridge. The threshold never
    goes below eigenmind's default (``SIMILARITY_THRESHOLD``).
    """
    n = len(C)
    if n <= 2:
        return SIMILARITY_THRESHOLD
    target_comps = min(min_components, max(2, n // 2))
    best_tau: float | None = None
    evaluated: set[float] = set()
    for q in candidates:
        tau = max(SIMILARITY_THRESHOLD, quantile_threshold(C, q))
        # Low-similarity collections clamp several candidates to the floor:
        # evaluate each distinct tau once.
        if tau in evaluated:
            continue
        evaluated.add(tau)
        _, labels = connected_components(threshold_graph(C, tau), directed=False)
        sizes = np.bincount(labels)
        n_comps = int(np.sum(sizes >= 2))
        max_size = int(np.max(sizes)) if len(sizes) > 0 else 0
        if n_comps >= target_comps and max_size <= int(max_dominant_ratio * n):
            logger.debug(
                "Adaptive threshold selected: tau=%.4f (q=%.2f, comps=%d, max_comp=%d/%d)",
                tau, q, n_comps, max_size, n,
            )
            return tau
        if n_comps >= target_comps and best_tau is None:
            best_tau = tau

    if best_tau is not None:
        return best_tau
    return max(SIMILARITY_THRESHOLD, quantile_threshold(C, 0.88))


def _query_utilities(graph: SimilarityGraph, query_vector: list[float]) -> np.ndarray | None:
    """Cosine of each chunk to the query, clipped at 0, as Supported Diversity utilities.

    Without them eigenmind uses unit utilities, i.e. a pure cardinality
    criterion that ignores the question. ``None`` (unit utilities) for an
    empty graph or a zero query vector.
    """
    q = np.asarray(query_vector, dtype=float)
    norm = np.linalg.norm(q)
    if graph.n == 0 or norm == 0:
        return None
    return np.clip(graph.embedding_matrix @ (q / norm), 0.0, None)


def _plot_laplacian_spectrum(graph: SimilarityGraph, output_filename_base: str) -> str:
    """Save the first 20 eigenvalues of the normalized Laplacian as a PNG; return its path.

    Replaces ``SimilarityGraph.eigenvalue_analysis``, removed in eigenmind 0.2.
    Uses a bare ``Figure`` rather than pyplot, whose global state is not
    thread-safe under Streamlit.
    """
    eigenvalues = np.linalg.eigvalsh(graph.laplacian())
    zero_multiplicity = int(np.sum(eigenvalues < 1e-3))
    n_to_plot = min(20, len(eigenvalues))
    xs = range(1, n_to_plot + 1)

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.plot(xs, eigenvalues[:n_to_plot], "o-")
    ax.set_title("Eigenvalues of the Graph Laplacian (Spectrum)")
    ax.set_xlabel(f"Eigenvalue Index (Sorted) - First {zero_multiplicity} are near-zero")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(list(xs))
    ax.grid(True)
    path = f"{output_filename_base}_eigenvalues.png"
    fig.savefig(path)
    return path


class GraphExplorer:
    """Build the artifacts shown by the Graph Explorer page (HTMLs, plot, ranking, Excel)."""

    def __init__(
        self,
        nlp,
        store: QdrantStore | None = None,
        embedder: EmbeddingModel | None = None,
        device: str | None = None,
    ):
        self.nlp = nlp
        self.store = store or QdrantStore()
        self._owns_embedder = embedder is None
        self.embedder = embedder or EmbeddingModel(device=device)

    def explore(
        self,
        collection_name: str,
        prompt: str,
        output_dir: str = "temp_graph_outputs",
        render_artifacts: bool = True,
    ) -> ExplorerArtifacts:
        """Run the graph analysis and (by default) render its HTML/Excel artifacts.

        Pass ``render_artifacts=False`` to skip the two pyvis HTML renders and
        the Excel export — only ``ranked_chunks``/``summary_text`` are computed.
        Used by the Chat page's hybrid RAG retrieval (:mod:`cogmaps.rag.pipeline`),
        which only reads ``ranked_chunks`` and has no use for a full pyvis render
        (with 1000 physics-stabilization iterations) or an Excel export on every
        single question.
        """
        logger.info("Graph explore collection=%r prompt=%r", collection_name, prompt[:80])
        os.makedirs(output_dir, exist_ok=True)

        try:
            query_vector = self.embedder.encode_query(prompt).tolist()
        finally:
            # The embedder is only needed for the query: free it before the
            # graph work below, whatever happens there.
            if self._owns_embedder:
                self.embedder.release()
        retrieved, initial_point_ids = explore_graph_with_initial_set(
            collection_name, query_vector, self.store.client, MAX_CHUNKS,
        )
        tau = SIMILARITY_THRESHOLD
        if retrieved:
            # C is computed again inside SimilarityGraph, but n <= MAX_CHUNKS
            # keeps that cheap, and it avoids patching the graph's internals.
            vectors = np.array([p.vector for p in retrieved], dtype=float)
            tau = select_adaptive_threshold(cosine_matrix(normalize_embeddings(vectors)))
        graph = SimilarityGraph(retrieved, threshold=tau)
        # Components / isolates show when the threshold degenerates (one
        # near-complete component, or many isolates that no method can select).
        # summary() fails on an empty graph.
        if graph.n > 0 and logger.isEnabledFor(logging.INFO):
            logger.info("Similarity graph tau=%.4f %s", graph.threshold, graph.summary())

        # 1. Main graph HTML
        main_html, document_colors = (
            self._render_main_graph(graph, prompt, initial_point_ids, output_dir)
            if render_artifacts else ("", {})
        )

        # 2. Eigenmind 0.2 selection mechanisms
        spec_result = graph.spectral_attention()
        spectral_indices = {c["index"] for c in spec_result["selected_chunks"]}

        # Default min_component_size (2): isolated chunks are noise to the
        # graph, and bridging them would rank off-topic chunks first.
        bridge_result = graph.semantic_bridges()
        bridge_indices = {c["index"] for c in bridge_result["selected_chunks"]}

        try:
            div_result = graph.supported_diversity(
                utilities=_query_utilities(graph, query_vector),
                budget=DIVERSITY_BUDGET,
                time_limit=DIVERSITY_TIME_LIMIT,
            )
            diversity_indices = set(div_result["selected_indices"])
        except RuntimeError:
            # The LP/MILP solver can fail numerically; the other two methods
            # still give a usable ranking.
            logger.warning("Supported Diversity failed; skipping it.", exc_info=True)
            diversity_indices = set()

        # Eigenvalue plot (generated when render_artifacts=True)
        eig_path = ""
        if render_artifacts and graph.n > 0:
            base = os.path.join(output_dir, _generate_filename_from_prompt(prompt, ""))
            try:
                eig_path = _plot_laplacian_spectrum(graph, base)
            except Exception:
                logger.warning("Eigenvalue plot failed; skipping it.", exc_info=True)

        # 3. Methods graph HTML
        methods_html = (
            self._render_methods_graph(
                graph, prompt, initial_point_ids,
                spectral_indices, bridge_indices, diversity_indices,
                output_dir,
            )
            if render_artifacts else ""
        )

        # 4. Consensus ranking
        ranked_chunks = self._consensus_ranking(
            graph, spectral_indices, bridge_indices, diversity_indices
        )
        summary = self._format_summary(ranked_chunks)

        # 5. Excel export
        excel_path = (
            self._export_excel(
                graph, spectral_indices, bridge_indices, diversity_indices, prompt, output_dir
            )
            if render_artifacts else ""
        )

        return ExplorerArtifacts(
            graph_html=main_html,
            eigenvalue_png=eig_path,
            summary_text=summary,
            methods_html=methods_html,
            excel_path=excel_path,
            ranked_chunks=ranked_chunks,
            document_colors=document_colors,
        )

    # ─── private rendering helpers ─────────────────────────────────

    def _render_main_graph(
        self, graph: SimilarityGraph, prompt: str, initial_ids: set, output_dir: str,
    ) -> tuple[str, dict[str, str]]:
        path = os.path.join(output_dir, _generate_filename_from_prompt(prompt, ".html"))
        net = Network(height="900px", width="100%", notebook=False, directed=False,
                      bgcolor=GRAPH_BG_COLOR, font_color=GRAPH_TEXT_COLOR)
        prompt_node_id = -1
        # pyvis renders `title` as HTML in the node tooltip — escape both the
        # user's prompt and the corpus text below, or a crafted document could
        # execute JS for anyone who hovers the node (stored XSS).
        net.add_node(prompt_node_id, label="PROMPT", title=html.escape(prompt), shape="box",
                     color=PROMPT_NODE_COLOR,
                     font={"color": GRAPH_PROMPT_TEXT_COLOR, "size": 28},
                     widthConstraint={"minimum": 140}, heightConstraint={"minimum": 60})

        unique_files = sorted({p.payload.get("filename") for p in graph.points})
        file_color = {fn: DOCUMENT_COLORS[i % len(DOCUMENT_COLORS)] for i, fn in enumerate(unique_files)}
        degrees = np.sum(graph.W > 0, axis=1)
        sizes = _scale_node_sizes(degrees)

        node_ids, labels, titles, colors, prompt_edges = [], [], [], [], []
        for i, pid in enumerate(graph.ordered_ids):
            p = graph.id_to_point[pid]
            node_ids.append(i)
            labels.append(str(i))
            safe_filename = html.escape(str(p.payload.get("filename", "N/A")))
            safe_text = html.escape(str(p.payload.get("text", "N/A")))
            titles.append(f"File: {safe_filename}\n\n{safe_text}")
            colors.append(file_color.get(p.payload.get("filename"), "#97c2fc"))
            if pid in initial_ids:
                prompt_edges.append((prompt_node_id, i))
        net.add_nodes(node_ids, label=labels, title=titles, color=colors, size=sizes)

        # Prune for visualization: keep top 5 neighbors per node
        n = graph.n
        vis_mask = np.zeros_like(graph.W, dtype=bool)
        for i in range(n):
            nbrs = np.where(graph.W[i] > 0)[0]
            if len(nbrs) > 5:
                top = nbrs[np.argsort(graph.W[i, nbrs])[-5:]]
                vis_mask[i, top] = True
            else:
                vis_mask[i, nbrs] = True
        vis_mask = vis_mask | vis_mask.T
        W_vis = np.where(vis_mask, graph.W, 0)

        net.set_options(json.dumps(_PHYSICS))
        rows, cols = np.where(np.triu(W_vis) > 0)
        edges = list(prompt_edges)
        for r, c in zip(rows, cols):
            edges.append((int(r), int(c), float(W_vis[r, c])))
        net.add_edges(edges)
        net.save_graph(path)
        _freeze_physics_after_stabilization(path)
        return path, file_color

    def _render_methods_graph(
        self, graph: SimilarityGraph, prompt: str, initial_ids: set,
        spectral_indices: set[int], bridge_indices: set[int], diversity_indices: set[int],
        output_dir: str,
    ) -> str:
        path = os.path.join(output_dir, _generate_filename_from_prompt(prompt, "_methods.html"))
        net = Network(height="900px", width="100%", notebook=False, directed=False,
                      bgcolor=GRAPH_BG_COLOR, font_color=GRAPH_TEXT_COLOR)
        prompt_node_id = -1
        net.add_node(prompt_node_id, label="PROMPT", title=html.escape(prompt), shape="box",
                     color=PROMPT_NODE_COLOR,
                     font={"color": GRAPH_PROMPT_TEXT_COLOR, "size": 28},
                     widthConstraint={"minimum": 140}, heightConstraint={"minimum": 60})

        relevant = spectral_indices | bridge_indices | diversity_indices
        ordered_relevant = sorted(relevant)

        # Prune for visualization: keep top 3 neighbors per node, same idea as
        # _render_main_graph. "Relevant" nodes are pre-selected as the most
        # central/connected ones, so an unpruned graph would link nearly every
        # pair, rendering as one illegible clump.
        vis_edges: dict[tuple[int, int], float] = {}
        for idx in ordered_relevant:
            sims = [(other, graph.W[idx, other]) for other in ordered_relevant
                    if other != idx and graph.W[idx, other] > 0]
            sims.sort(key=lambda x: x[1], reverse=True)
            for other, sim in sims[:3]:
                key = (idx, other) if idx < other else (other, idx)
                vis_edges[key] = max(vis_edges.get(key, 0.0), float(sim))

        degrees = np.array([
            sum(1 for (a, b) in vis_edges if a == idx or b == idx)
            for idx in ordered_relevant
        ])
        sizes = dict(zip(ordered_relevant, _scale_node_sizes(degrees)))

        for idx in ordered_relevant:
            p = graph.id_to_point[graph.ordered_ids[idx]]
            if idx in spectral_indices: color = METHOD_COLORS["spectral_attention"]
            elif idx in bridge_indices: color = METHOD_COLORS["semantic_bridge"]
            elif idx in diversity_indices: color = METHOD_COLORS["supported_diversity"]
            else: color = "#97c2fc"
            methods = []
            if idx in spectral_indices: methods.append("spectral_attention")
            if idx in bridge_indices: methods.append("semantic_bridge")
            if idx in diversity_indices: methods.append("supported_diversity")
            # See _render_main_graph: pyvis titles render as HTML, so corpus
            # text must be escaped to avoid stored XSS.
            safe_filename = html.escape(str(p.payload.get("filename", "N/A")))
            safe_text = html.escape(str(p.payload.get("text", "N/A")))
            title = (f"File: {safe_filename}\n"
                     f"Method: {', '.join(methods)}\n\n{safe_text}")
            net.add_node(int(idx), label=str(idx), title=title, color=color, size=sizes[idx])

        for idx in ordered_relevant:
            if graph.ordered_ids[idx] in initial_ids:
                net.add_edge(prompt_node_id, int(idx))
        for (a, b), sim in vis_edges.items():
            net.add_edge(int(a), int(b), value=sim)
        net.set_options(json.dumps(_METHODS_PHYSICS))
        net.save_graph(path)
        _freeze_physics_after_stabilization(path)
        return path

    def _consensus_ranking(
        self, graph: SimilarityGraph,
        spectral_indices: set[int], bridge_indices: set[int], diversity_indices: set[int],
    ) -> list[dict]:
        relevant = spectral_indices | bridge_indices | diversity_indices
        out = []
        for idx in relevant:
            p = graph.id_to_point[graph.ordered_ids[idx]]
            methods_display = []
            if idx in spectral_indices: methods_display.append("spectral_attention")
            if idx in bridge_indices: methods_display.append("semantic_bridge")
            if idx in diversity_indices: methods_display.append("supported_diversity")
            out.append({
                "filename": p.payload.get("filename", "N/A"),
                "text": p.payload.get("text", "N/A"),
                "chunk_id": p.payload.get("chunk_number", "N/A"),
                "methods": methods_display,
                "count": len(methods_display),
                "themes": _extract_themes(p.payload.get("text", ""), self.nlp),
            })
        out.sort(
            key=lambda c: (
                c["count"],
                "semantic_bridge" in c["methods"],
                "supported_diversity" in c["methods"],
            ),
            reverse=True,
        )
        return out

    @staticmethod
    def _format_summary(ranked_chunks: list[dict]) -> str:
        summary = "--- Consolidated Ranking (by Method Consensus) ---\n"
        for i, c in enumerate(ranked_chunks):
            text = " ".join(c["text"].split())
            wrapped = textwrap.fill(text, width=120, initial_indent=" " * 6, subsequent_indent=" " * 6)
            summary += f"\n\n[{i + 1}] Consensus Score: {c['count']}/3 | Methods: {', '.join(c['methods'])}"
            summary += f"\n    Themes: {', '.join(c['themes'])}"
            summary += f"\n    - [{c['chunk_id']} | {c['filename']}]"
            summary += f'\n      "{wrapped.strip()}"'
        return summary

    @staticmethod
    def _export_excel(
        graph: SimilarityGraph,
        spectral_indices: set[int], bridge_indices: set[int], diversity_indices: set[int],
        prompt: str, output_dir: str,
    ) -> str:
        path = os.path.join(output_dir, _generate_filename_from_prompt(prompt, ".xlsx"))
        node_rows = []
        for idx, pid in enumerate(graph.ordered_ids):
            p = graph.id_to_point[pid]
            methods = []
            if idx in spectral_indices: methods.append("spectral_attention")
            if idx in bridge_indices: methods.append("semantic_bridge")
            if idx in diversity_indices: methods.append("supported_diversity")
            sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", str(p.payload.get("text", "")))
            node_rows.append({
                "Node ID": idx,
                "Chunk Number": p.payload.get("chunk_number", "N/A"),
                "Source File": p.payload.get("filename", "N/A"),
                "Selected Methods": ", ".join(methods) if methods else "None",
                "Chunk Text": sanitized,
            })
        edge_rows = []
        rows, cols = np.where(np.triu(graph.W) > 0)
        for r, c in zip(rows, cols):
            edge_rows.append({"Source Node": int(r), "Target Node": int(c), "Weight": float(graph.W[r, c])})

        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(node_rows).to_excel(writer, sheet_name="Nodes", index=False)
            pd.DataFrame(edge_rows).to_excel(writer, sheet_name="Adjacency Matrix", index=False)
        return path
