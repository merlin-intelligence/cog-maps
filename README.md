The software is licensed under MIT.

The CogMaps and Merlin Intelligence names, logos and branding are not covered by the MIT license and may not be used without permission.
# cog-maps - /accelerate clarity/

CogMaps is a sophisticated knowledge management and exploration application built with Streamlit. It leverages a local Vector Database (Qdrant via Docker) and advanced AI models to ingest, map, and query your proprietary document corpus. LLM generation supports two interchangeable backends: **Nebius AI** (cloud, Llama / Kimi / OSS models) and **Ollama** (fully local, no data leaves the machine).

## Core Features

1. **Multi-Source Ingestion (`/add files to your corpus/`)**
   - Local directories (PDF, Word, Excel, CSV, PowerPoint, Text, Markdown) with OCR support — each format is parsed to markdown (PDF/DOCX/XLSX/CSV via ChunkNorris, PowerPoint via MarkItDown) and chunked uniformly by ChunkNorris. JSON is not supported yet.
   - Direct synchronization with Google Drive (OAuth or Service Account).
   - Direct synchronization with Microsoft SharePoint.
   - **Browser-resilient async ingestion**: jobs run in a background daemon thread decoupled from the browser session, so tab closure, network interruption, or browser timeout does not kill the process. Job state (status, progress, logs) is persisted to SQLite (`user_data/jobs.db`) and the UI reconnects automatically to a running job on page reload.
   - **Smart Resume**: File-level resume feature using Qdrant filename tracking to skip already processed documents.
   - Local multilingual embedding (`intfloat/multilingual-e5-base`, 768-dim) via `sentence-transformers`, with the E5 `query:` / `passage:` prefixes applied automatically.

2. **Knowledge Graph Navigation (`/navigate your experience/`)**
   - Generates interactive subgraphs of knowledge relationships based on specific prompts.
   - Performs eigenvector/Laplacian analysis and identifies **Singular**, **Hinge**, and **Theta** nodes for unique, non-obvious insights.

3. **Advanced Question Answering (`/ask/`)**
   - Hybrid Retrieval-Augmented Generation (RAG) using both standard semantic similarity and singular chunk analysis.
   - **Nebius AI** backend: Llama 3.3, Kimi 2.5, OpenAI OSS 120B via cloud API (requires `NEBIUS_API_KEY`).
   - **Ollama** backend: any locally installed model (e.g. `qwen2.5:7b`), fully on-premise. Switch between backends from the sidebar toggle.

4. **Corpus Analysis (`/analyze corpus/`)**
   - Document inventory by type, character-length distribution, and a TF-IDF-stopworded wordcloud.
   - **Embedding-based topic modeling**: KMeans on the mean per-document embedding (reused from Qdrant — no re-encoding), `k` chosen by silhouette (cosine) over `[2..20]`. Cluster keywords are extracted *post-hoc* by TF-IDF so the partition stays semantic while the labels stay readable.
   - 2D PCA map coloured by cluster + pie-chart of cluster distribution.
   - Near-duplicate detection: only document pairs with cosine similarity ≥ 0.99 are surfaced, each flagged as **near-duplicate** and ranked by descending score.

5. **Ontology Building & Exploration (`/build ontology/`, `/explore ontology/`)**
   - Builds an OWL/RDFS ontology from documents already in Qdrant: a Nebius tool-calling agent drives [OLAF](https://github.com/merlin-intelligence/olaf) (MCP server) and stores the triples in **Oxigraph** (RDF triplestore, via Docker).
   - Background jobs, like ingestion — an interrupted build can be re-launched and resumes where it stopped (processed chunks are skipped).
   - Optional **domain discovery**: a per-collection *domain blueprint* (inferred domain + taxonomical pillars), synthesized from a corpus sample or written by hand, tailors the agent's prompt. Opt-in per build.
   - **Ask the ontology**: a read-only search agent (ported from OLAF's `olaf_searching_agent` demo) answers natural-language questions with OLAF's search tools and agent-written SPARQL, then shows the part of the ontology it used (entities + direct neighbors) and the source chunks behind the answer. Follow-up questions keep the conversation's context.
   - Read-only explorer: browse an ontology as a graph or raw Turtle, and run SPARQL queries.

## Stability & Performance

- **Shared Model Cache**: The embedding model is loaded once on first use, kept resident via `@st.cache_resource` — a single ~300 MB copy is reused across all sessions **and background ingestion jobs** (no per-job reload), with concurrent calls serialized internally so ingestion and chat/analysis never race for the same model instance.
- **CPU-First**: CPU-only PyTorch by default for broad compatibility and predictable memory usage; CUDA / MPS auto-detected when available.
- **Cold-Start Buffer**: On 4 GB-RAM VMs a 3.3 GB swap file is recommended to absorb the first-load spike (model download + load).
- **Persistent Storage**: Vector embeddings are stored persistently in the named Docker volume `qdrant_storage` (see `docker-compose.yml`) — not a plain directory in the repo. Job state and logs are persisted in `user_data/jobs.db`.

## Getting Started

Please refer to the [Architecture and Installation Guide](docs/ARCHITECTURE_AND_INSTALLATION_GUIDE.md) for detailed instructions on deploying the Qdrant database, installing dependencies, and configuring API keys.

## Project layout

```
cogmaps/                      importable package
├── config.py                 single source of truth for constants and env-based secrets
├── qdrant/                   Qdrant management — client, ingestion, retrieval, deletion (QdrantStore)
├── rag/                      RAG answer-generation — hybrid retrieval + LLM call (isolated
│                             from the Chat page), Nebius/Ollama chat clients
├── graph/                    graph visualization — Qdrant-based BFS exploration + the
│                             GraphExplorer pipeline (HTML graphs, eigenvalue plot, Excel
│                             export); graph math itself lives in the `eigenmind` package
│                             (see below)
├── core/                     embeddings, chunking, corpus analysis
├── connectors/                Google Drive, SharePoint
├── pipelines/                 ingestion orchestration (Ingester)
├── jobs/                       async ingestion job system (SQLite store + daemon runner)
├── ontology/                   ontology building — OLAF/MCP agent loop, build job runner,
│                             domain discovery, search agent, Oxigraph client, graph view, cleanup
└── ui/                          Streamlit-only code (auth, components, styles)

streamlit_app.py            entry point — `streamlit run streamlit_app.py`
pages/                      Streamlit multipage navigation (/add files to your corpus/, /analyze corpus/, /ask/, /navigate your experience/, /explore ontology/, /build ontology/, /manage/)
scripts/ingest_recursive.py CLI ingestion (also exposed as `cogmaps-ingest`)
tests/unit/                 unit tests (app-level; pure graph-math tests live in the eigenmind package)
```

The graph algorithms themselves (singular chunks, ℓ∞-connectivity/hinge ranking,
Lovász θ diversity, `SimilarityGraph`) are published separately as the
[`eigenmind`](https://github.com/merlin-intelligence/eigenmind) package and
pulled in as a regular dependency of `cogmaps/graph/` (see `pyproject.toml` / `requirements.txt`).

Run the app (`QDRANT_API_KEY` must be set in `.env` first):

```bash
pip install -e .
docker compose up -d          # Qdrant + Oxigraph
streamlit run streamlit_app.py
```

CLI ingestion:

```bash
cogmaps-ingest directories.txt my_collection --device cpu
```

Secrets come from `.env` or `.streamlit/secrets.toml` (see `.env.example`).
The HuggingFace token is **never** hard-coded — set `HF_TOKEN` in your env if
you need a gated local LLM.

## Contributing

Contributions are welcome! Please read the [Contributing Guide](docs/CONTRIBUTING.md) before opening an issue or a pull request — it covers the workflow, naming conventions, and a few basic rules that keep collaboration smooth.

---
© 2026 Merlin Intelligence
