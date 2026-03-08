"""
rise_rag/scripts/ingest.py — Build (or rebuild) the RISE RAG knowledge base.

Run from the project root (the directory that contains ``api.py``):

    python rise_rag/scripts/ingest.py

What this script does
---------------------
1. Reads every ``.txt`` file from ``rise_rag/data/``.
2. Parses each file into document chunks using the
   ``DOCUMENT / PARCEL_ID / TOPIC`` metadata headers.
3. Generates 384-dimensional embeddings locally with sentence-transformers
   (``all-MiniLM-L6-v2``).  No API key required.
4. Upserts all chunks into the persistent ChromaDB collection at
   ``rise_rag/chroma_db/``.

It is safe to re-run — existing records are updated in-place and no
duplicate chunks are created.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable regardless of the working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent          # rise_rag/scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # repo root
sys.path.insert(0, str(_PROJECT_ROOT))

from rise_rag.app.config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL
from rise_rag.app.embeddings import VectorStore
from rise_rag.app.ingestion import load_all_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rise.ingest")

_DIVIDER = "=" * 62


def main() -> None:
    print(f"\n{_DIVIDER}")
    print("  RISE Knowledge Base — Ingestion")
    print(_DIVIDER)

    # ── 1. Locate data files ──────────────────────────────────────────────────
    if not DATA_DIR.exists():
        logger.error("Data directory not found: %s", DATA_DIR)
        sys.exit(1)

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", DATA_DIR)
        sys.exit(1)

    print(f"\n  Source directory : {DATA_DIR}")
    print(f"  Files found      : {len(txt_files)}")
    for f in txt_files:
        print(f"    • {f.name}")

    # ── 2. Parse into chunks ──────────────────────────────────────────────────
    print("\n  Parsing documents …")
    try:
        chunks = load_all_chunks(DATA_DIR)
    except Exception:
        logger.exception("Failed to parse knowledge-base files.")
        sys.exit(1)

    print(f"  Total chunks     : {len(chunks)}")

    # Breakdown by parcel_id for a quick sanity check.
    parcel_counts: dict[str, int] = {}
    for chunk in chunks:
        parcel_counts[chunk.parcel_id] = parcel_counts.get(chunk.parcel_id, 0) + 1
    print("\n  Chunk breakdown by parcel_id:")
    for parcel_id, count in sorted(parcel_counts.items()):
        print(f"    {parcel_id:<12} {count} chunks")

    # ── 3. Initialise vector store ────────────────────────────────────────────
    print(f"\n  Embedding model  : {EMBEDDING_MODEL}")
    print(f"  ChromaDB path    : {CHROMA_PERSIST_DIR}")
    print("  (First run downloads ~80 MB from HuggingFace and caches it.)")

    store = VectorStore()
    existing = store.count()
    if existing > 0:
        print(f"  Existing records : {existing} — will upsert (no duplicates)")

    # ── 4. Upsert ─────────────────────────────────────────────────────────────
    print(f"\n  Upserting {len(chunks)} chunks into ChromaDB …")
    try:
        store.upsert_chunks(chunks, batch_size=50)
    except Exception:
        logger.exception("Ingestion failed during upsert.")
        sys.exit(1)

    print(f"\n  Done. Collection now contains {store.count()} chunks.")
    print(_DIVIDER + "\n")


if __name__ == "__main__":
    main()
