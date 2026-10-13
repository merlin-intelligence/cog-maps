from unittest.mock import MagicMock, patch
import numpy as np

from cogmaps.rag.pipeline import answer_question
from cogmaps.graph.explorer import ExplorerArtifacts
from cogmaps.ui.styles import badge


def test_answer_question_with_eigenmind_labels():
    store = MagicMock()
    embedder = MagicMock()
    embedder.encode_query.return_value = np.zeros(128)
    llm_client = MagicMock()
    llm_client.chat.return_value = "This is an answer [Chunk 1] [Chunk 2]."

    # Similarity search returns Chunk 1
    store.similarity_search.return_value = [
        {"text": "Text 1", "filename": "doc1.pdf", "chunk_id": 0}
    ]

    # Graph explorer returns Chunk 1 (overlapping) and Chunk 2 (distinct)
    fake_artifacts = ExplorerArtifacts(
        graph_html="",
        eigenvalue_png="",
        summary_text="Summary",
        methods_html="",
        excel_path="",
        ranked_chunks=[
            {
                "text": "Text 1",
                "filename": "doc1.pdf",
                "chunk_id": 0,
                "methods": ["spectral_attention"],
                "score": 1,
            },
            {
                "text": "Text 2",
                "filename": "doc2.pdf",
                "chunk_id": 1,
                "methods": ["semantic_bridge", "supported_diversity"],
                "score": 2,
            },
        ],
    )

    with patch("cogmaps.rag.pipeline.GraphExplorer") as mock_explorer_cls:
        mock_explorer = MagicMock()
        mock_explorer.explore.return_value = fake_artifacts
        mock_explorer_cls.return_value = mock_explorer

        result = answer_question(
            store=store,
            embedder=embedder,
            nlp=MagicMock(),
            llm_client=llm_client,
            collection="test_col",
            prompt="What is test?",
            num_similar=2,
            num_singular=2,
        )

    assert result.answer == "This is an answer [Chunk 1] [Chunk 2]."
    assert len(result.retrieved) == 2

    # Chunk 1 was in similarity search and graph exploration
    c1 = result.retrieved[0]
    assert c1["methods"] == ["spectral_attention"]
    assert "Similarity Search + Graph (spectral_attention)" in c1["source_type"]

    # Chunk 2 was in graph exploration only
    c2 = result.retrieved[1]
    assert c2["methods"] == ["semantic_bridge", "supported_diversity"]
    assert c2["source_type"] == "Graph (semantic_bridge, supported_diversity)"

    # Verify badge formatting
    b_spec = badge("spectral_attention")
    assert "badge-spectral-attention" in b_spec
    assert "spectral_attention" in b_spec

    b_bridge = badge("semantic_bridge")
    assert "badge-semantic-bridge" in b_bridge
    assert "semantic_bridge" in b_bridge

    b_div = badge("supported_diversity")
    assert "badge-supported-diversity" in b_div
    assert "supported_diversity" in b_div
