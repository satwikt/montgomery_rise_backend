"""
scripts/ingest.py — One-time ingestion script for RISE knowledge base.

Run this from the project root:
    python scripts/ingest.py

This will:
1. Read all .txt files from the data/ folder
2. Parse them into chunks with metadata
3. Generate embeddings using sentence-transformers (local, free)
4. Store everything in a local ChromaDB database

It is safe to re-run — existing chunks are upserted (not duplicated).
"""

import sys
import logging
from pathlib import Path

# Add project root to path so we can import rise_rag
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion import load_all_chunks
from app.embeddings import VectorStore
from app.config import DATA_DIR, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  RISE Knowledge Base Ingestion")
    print("=" * 60)

    # 1. Check data directory
    if not DATA_DIR.exists():
        print(f"\n Data directory not found: {DATA_DIR}")
        print("   Create a 'data/' folder and add your .txt files.")
        sys.exit(1)

    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"\n No .txt files found in {DATA_DIR}")
        print("   Copy your knowledge base files into the data/ folder.")
        sys.exit(1)

    print(f"\n Data directory: {DATA_DIR}")
    print(f"   Found {len(txt_files)} file(s):")
    for f in sorted(txt_files):
        print(f"   - {f.name}")

    # 2. Parse chunks
    print(f"\n Parsing documents...")
    chunks = load_all_chunks(DATA_DIR)
    print(f"  Parsed {len(chunks)} chunks from {len(txt_files)} files")

    # Show chunk breakdown by parcel
    parcel_counts: dict[str, int] = {}
    for chunk in chunks:
        pid = chunk.parcel_id
        parcel_counts[pid] = parcel_counts.get(pid, 0) + 1

    print("\n   Chunk breakdown by parcel_id:")
    for parcel, count in sorted(parcel_counts.items()):
        print(f"   - {parcel:10s}: {count} chunks")

    # 3. Load embedding model & create vector store
    print(f"\n Embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~80MB from HuggingFace, cached after)")
    print(f"   Initialising ChromaDB at: {CHROMA_PERSIST_DIR}")
    
    store = VectorStore()
    existing = store.count()
    if existing > 0:
        print(f"   Collection already has {existing} documents — will upsert")

    # 4. Upsert into ChromaDB
    print(f"\n Upserting {len(chunks)} chunks into ChromaDB...")
    total = store.upsert_chunks(chunks, batch_size=50)
    
    print(f"\n Ingestion complete!")
    print(f"   Total chunks in collection: {store.count()}")
    print(f"\n Run 'python chat.py' to start chatting.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
