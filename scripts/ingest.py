#!/usr/bin/env python3
"""
Ingestion entry-point script.

Run this once to:
  1. Clone all 6 algorithm repos into data/raw/
  2. Chunk every source file with language-aware splitting
  3. Embed every chunk via Cohere embed-english-v3.0
  4. Persist the vector store in data/chroma_db/

Usage:
    python scripts/ingest.py                  # full ingest
    python scripts/ingest.py --force          # wipe and re-embed everything

The pipeline is fully idempotent: already-indexed chunks are skipped
automatically. Safe to interrupt and re-run at any point.

Estimated time: 30–90 minutes on first run (Cohere free tier rate limit).
"""

import sys
import argparse
from pathlib import Path

# Make sure `src/` is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import settings
from src.ingestion.indexer import run_full_indexing_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Build the AlgoRAG vector knowledge base."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe the existing index and re-embed everything from scratch.",
    )
    args = parser.parse_args()

    if args.force:
        import shutil
        chroma_path = Path(settings.chroma_db_path)
        if chroma_path.exists():
            print(f"⚠️  --force: deleting existing index at {chroma_path}")
            shutil.rmtree(chroma_path)

    indexer = run_full_indexing_pipeline(settings)

    print("\n🎉 Ingestion complete. Run the app with:")
    print("   python -m src.ui.app\n")

    return indexer


if __name__ == "__main__":
    main()
