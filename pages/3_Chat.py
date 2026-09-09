"""Chat page — hybrid RAG (similarity search + graph) over a Qdrant collection."""
from __future__ import annotations

import html
import logging
import re

import streamlit as st

from cogmaps.qdrant.store import QdrantStore
from cogmaps.rag.llm_clients import build_llm_client
from cogmaps.rag.pipeline import answer_question
from cogmaps.ui.auth import (
    check_password,
    list_visible_collections,
)
from cogmaps.ui.components import (
    empty_state,
    get_embedder,
    load_nlp,
    render_sidebar,
    section_header,
)
from cogmaps.ui.styles import apply_global_styles, badge

logger = logging.getLogger(__name__)

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/ask/", "hybrid RAG · graph + similarity")

if not sb.is_connected:
    empty_state("⚡", "Qdrant offline.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
col_labels = [label for label, _ in _visible]
col_map = {label: qdrant_name for label, qdrant_name in _visible}

if not col_labels:
    empty_state("📂", "No collections found. Ingest documents in <strong>/enrich corpus/</strong> first.")
    st.stop()

col_q1, col_q2 = st.columns([1, 2])
with col_q1:
    collection_name = st.selectbox("collection", col_labels)
    qdrant_col = col_map[collection_name]
    pt_count, _ = store.collection_stats(qdrant_col)
    if pt_count:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
            f'{pt_count:,} vectors</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
        'color:#8a6a50;margin-top:1rem">retrieval settings</p>',
        unsafe_allow_html=True,
    )
    num_similar = st.slider("similarity chunks", 0, 20, 5)
    num_singular = st.slider("singular chunks", 0, 20, 5)

    short_model = sb.llm_model.split("/")[-1] if sb.llm_model else "—"
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#8a6a50">'
        f'model: {short_model} · {sb.llm_provider}</p>',
        unsafe_allow_html=True,
    )

with col_q2:
    prompt = st.text_area(
        "your question",
        "what is intellectual humility?",
        height=120,
        help="Ask anything about your corpus. The system blends semantic search with graph analysis.",
    )

    not_ready_msg = (
        "⚠ Ollama is offline or no model is selected. Start the server "
        "(<code>ollama serve</code>) and pull a model (<code>ollama pull qwen2.5:7b</code>)."
        if sb.llm_provider == "ollama"
        else "⚠ AI Hub API key is not configured. Please contact your administrator."
    )
    if not sb.llm_ready:
        st.markdown(
            f'<div class="info-box" style="border-left-color:#a82020;background:#f5e8e8">'
            f'{not_ready_msg}</div>',
            unsafe_allow_html=True,
        )

    ask_btn = st.button("▶ get answer", type="primary", disabled=(not sb.llm_ready))


def _run_query() -> None:
    with st.spinner("retrieving context and generating…"):
        try:
            qdrant_col = col_map[collection_name]
            embedder = get_embedder(sb.selected_device)
            llm = build_llm_client(sb)

            result = answer_question(
                store=store,
                embedder=embedder,
                nlp=load_nlp(),
                llm_client=llm,
                collection=qdrant_col,
                prompt=prompt,
                num_similar=num_similar,
                num_singular=num_singular,
            )
            answer = result.answer
            retrieved = result.retrieved

            st.markdown("---")
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
                'margin-bottom:0.5rem">⬡ answer</p>',
                unsafe_allow_html=True,
            )
            # The answer is LLM-generated text that may echo/quote corpus content —
            # escape before embedding in raw HTML to avoid stored XSS via a
            # crafted document that gets cited back verbatim.
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #c0b4a8;border-left:3px solid #c44a28;'
                f'border-radius:10px;padding:1.4rem 1.6rem;font-size:0.95rem;line-height:1.75;color:#2a1f18;'
                f'white-space:pre-wrap">'
                f'{html.escape(answer)}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                'letter-spacing:0.15em;color:#8a6a50;text-transform:uppercase;'
                'margin-bottom:0.5rem">references</p>',
                unsafe_allow_html=True,
            )
            for i, c in enumerate(retrieved):
                src = c["source_type"]
                is_graph = any(k in src for k in ("Graph", "Singular", "Hinge", "Theta"))
                badge_html = badge("graph" if is_graph else "sim")
                with st.expander(f"[{i + 1}] {c['filename']} · chunk {c['chunk_number']}"):
                    st.markdown(
                        f'<p style="margin-bottom:0.5rem">{badge_html} '
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                        f'color:#8a6a50;margin-left:0.5rem">{html.escape(src)}</span></p>',
                        unsafe_allow_html=True,
                    )
                    st.write(c["text"])

            content_parts = [
                f"Question: {prompt}\n\n{'=' * 40}\nLLM Answer:\n{'=' * 40}\n\n{answer}\n\n"
                f"{'=' * 40}\nReferences:\n{'=' * 40}\n\n"
            ]
            for i, c in enumerate(retrieved):
                content_parts.append(
                    f"[{i + 1}] {c['filename']} (Chunk {c['chunk_number']}) — {c['source_type']}\n"
                    f"---\n{c['text']}\n---\n\n"
                )
            sanitized = re.sub(r"[^\w\s-]", "", prompt).strip().replace(" ", "_")
            st.download_button(
                "⬇ download answer + references",
                "".join(content_parts).encode("utf-8"),
                file_name=f"answer_{sanitized[:30]}.txt",
                mime="text/plain",
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to generate answer: {e}")
            logger.exception("Answer generation failed")


if ask_btn and collection_name and prompt and sb.llm_ready:
    _run_query()
elif ask_btn and not sb.llm_ready:
    st.warning(
        "Ollama is offline or no model is selected."
        if sb.llm_provider == "ollama"
        else "AI Hub API key is not configured. Please contact your administrator."
    )
elif ask_btn:
    st.warning("Provide a collection and a question.")
