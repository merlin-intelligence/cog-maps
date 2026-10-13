"""Hybrid RAG answer-generation pipeline: similarity + graph retrieval, then an LLM call."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

from cogmaps.config import MAX_CONTEXT_CHARS
from cogmaps.core.embeddings import EmbeddingModel
from cogmaps.graph.explorer import GraphExplorer
from cogmaps.qdrant.store import QdrantStore


@dataclass
class AnswerResult:
    """Result of :func:`answer_question`: the LLM answer plus the chunks used as context."""
    answer: str
    retrieved: list[dict] = field(default_factory=list)


def answer_question(
    store: QdrantStore,
    embedder: EmbeddingModel,
    nlp,
    llm_client,
    collection: str,
    prompt: str,
    num_similar: int,
    num_singular: int,
) -> AnswerResult:
    """Retrieve context (similarity search + graph exploration) and generate an answer.

    Blends plain cosine-similarity retrieval with the Graph Explorer's ranked "singular"
    chunks (eigenvector/hinge/theta consensus), then asks ``llm_client`` to answer
    ``prompt`` citing chunk numbers. ``llm_client`` exposes ``chat(system_prompt, user_content)``
    (see :mod:`cogmaps.rag.llm_clients`).
    """
    # 1. Similarity retrieval
    query_vec = embedder.encode_query(prompt).tolist()
    sim_chunks = store.similarity_search(collection, query_vec, limit=num_similar)
    for c in sim_chunks:
        c["source_type"] = "Similarity Search"
    retrieved = list(sim_chunks)

    # 2. Graph retrieval (spectral_attention/semantic_bridge/supported_diversity)
    # — only ranked_chunks is used below, so skip pyvis HTML/Excel rendering
    # (render_artifacts=False) that the Graph Explorer page needs but a chat answer doesn't.
    with tempfile.TemporaryDirectory() as tmp_dir:
        explorer = GraphExplorer(nlp, store=store, embedder=embedder)
        artifacts = explorer.explore(collection, prompt, output_dir=tmp_dir, render_artifacts=False)

        added = 0
        for ch in artifacts.ranked_chunks:
            if added >= num_singular:
                break
            existing = next((c for c in retrieved if c["text"] == ch["text"]), None)
            if existing is not None:
                existing_methods = list(existing.get("methods") or [])
                for m in ch.get("methods", []):
                    if m not in existing_methods:
                        existing_methods.append(m)
                existing["methods"] = existing_methods
                methods_str = ", ".join(existing_methods)
                existing["source_type"] = (
                    f"Similarity Search + Graph ({methods_str})"
                    if methods_str
                    else "Similarity Search + Graph"
                )
                added += 1
            else:
                methods_str = ", ".join(ch.get("methods", []))
                retrieved.append({
                    "text": ch["text"],
                    "filename": ch["filename"],
                    "chunk_number": ch["chunk_id"],
                    "methods": ch.get("methods", []),
                    "source_type": f"Graph ({methods_str})" if methods_str else "Graph",
                    "score": None,
                })
                added += 1

    # 3. Build context (capped so a generous slider selection can't silently
    # blow past the model's context window), call LLM
    context = ""
    for i, c in enumerate(retrieved):
        piece = f"--- Chunk {i + 1} ({c['source_type']}) ---\n{c['text']}\n\n"
        if len(context) + len(piece) > MAX_CONTEXT_CHARS:
            context += "[... remaining retrieved chunks omitted to stay within the context limit ...]\n"
            break
        context += piece

    answer = llm_client.chat(
        system_prompt=(
            "You are a helpful assistant. Answer the question based on the provided context. "
            "Explicitly cite the chunk number (e.g. [Chunk 1]) for every piece of information you use. "
            "Reply with the final answer only: do not describe your reasoning, your analysis of the "
            "chunks or the steps you followed."
        ),
        user_content=f"Context:\n{context}\n\nQuestion: {prompt}",
    )

    return AnswerResult(answer=answer, retrieved=retrieved)
