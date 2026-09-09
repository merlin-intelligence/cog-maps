"""CLI entry point for recursive ChunkNorris ingestion.

Usage:
    python -m scripts.ingest_recursive directories.txt my_collection

Or, after `pip install -e .`:
    mi-eigenmind-ingest directories.txt my_collection
"""
from __future__ import annotations

import argparse
import sys

from cogmaps.pipelines.ingest import Ingester
from cogmaps.qdrant.store import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively ingest PDFs from a list of directories into Qdrant.",
    )
    parser.add_argument("directories_file", help="Path to a .txt file with one directory per line.")
    parser.add_argument("collection_name", help="Qdrant collection name (will be created if missing).")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detected if omitted)")
    parser.add_argument("--host", default=None, help="Qdrant host (defaults to QDRANT_HOST env)")
    parser.add_argument("--port", type=int, default=None, help="Qdrant port (defaults to QDRANT_PORT env)")
    args = parser.parse_args()

    store = QdrantStore(host=args.host, port=args.port)
    ingester = Ingester(store=store, device=args.device)
    for line in ingester.run_chunknorris(args.directories_file, args.collection_name):
        print(line)
    if ingester.files_attempted > 0 and ingester.files_failed == ingester.files_attempted:
        print(f"All {ingester.files_attempted} file(s) failed to process.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
