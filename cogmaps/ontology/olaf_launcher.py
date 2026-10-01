"""Entry point that starts OLAF with cog-maps' embedding model available to fastembed.

OLAF embeds concepts with fastembed (``olaf.embeddings.EmbeddingService``),
which only loads models from its own catalog — and
``intfloat/multilingual-e5-base`` (:data:`cogmaps.config.EMBEDDING_MODEL_NAME`)
is not in it. fastembed can load any ONNX export through
``TextEmbedding.add_custom_model``, and the model's Hugging Face repo ships
one (``onnx/model.onnx``), so this module registers it and then hands over
to OLAF's own ``main()``. Run as ``python -m cogmaps.ontology.olaf_launcher``
with the same arguments as the ``olaf`` console script.

Nothing here may write to stdout: in server mode it carries the MCP stdio
protocol.
"""
from __future__ import annotations

import os
from pathlib import Path

from cogmaps.config import EMBEDDING_DIM_DEFAULT, EMBEDDING_MODEL_NAME


def register_embedding_model() -> None:
    """Make :data:`EMBEDDING_MODEL_NAME` loadable by ``fastembed.TextEmbedding`` (idempotent)."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    supported = {m["model"].lower() for m in TextEmbedding.list_supported_models()}
    if EMBEDDING_MODEL_NAME.lower() in supported:
        return
    # Same weights and post-processing as sentence-transformers' E5 config:
    # mean pooling over tokens, then L2 normalization.
    TextEmbedding.add_custom_model(
        model=EMBEDDING_MODEL_NAME,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=EMBEDDING_MODEL_NAME),
        dim=EMBEDDING_DIM_DEFAULT,
        model_file="onnx/model.onnx",
    )


def main() -> None:
    # fastembed's default cache is under /tmp, wiped on reboot — the ~1.1 GB
    # model would be downloaded again. FASTEMBED_CACHE_PATH still wins if set.
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
    register_embedding_model()
    from olaf.server import main as olaf_main

    olaf_main()


if __name__ == "__main__":
    main()
