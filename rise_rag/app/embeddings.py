"""
embeddings.py — ChromaDB vector store wrapper for RISE RAG.

Handles:
- Creating/opening the ChromaDB collection
- Embedding chunks with sentence-transformers (local, free, no API key)
- Upserting chunks (safe to re-run ingestion)
- Querying for similar chunks
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Collection

from .config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    MAX_DISTANCE_THRESHOLD,
)
from .ingestion import Chunk

logger = logging.getLogger(__name__)


# ─── Embedding Function ───────────────────────────────────────────────────────

def _get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """
    Returns a ChromaDB-compatible embedding function using sentence-transformers.
    
    all-MiniLM-L6-v2:
      - 384-dimensional vectors
      - ~80MB download, cached locally after first run
      - Fast inference, good semantic quality for retrieval tasks
      - Completely free, no API key
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


# ─── VectorStore Class ────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB for RISE knowledge base storage and retrieval.
    
    Usage:
        store = VectorStore()
        store.upsert_chunks(chunks)
        results = store.query("What is the heritage score for Parcel A?")
    """

    def __init__(self) -> None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR)
        )
        self._embed_fn = _get_embedding_function()
        self._collection: Collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "l2"},  # L2 distance (Euclidean)
        )
        logger.info(
            "VectorStore ready. Collection '%s' has %d documents.",
            CHROMA_COLLECTION,
            self._collection.count(),
        )

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> int:
        """
        Upsert a list of Chunks into ChromaDB.
        
        Upsert = insert or update if the chunk_id already exists.
        Safe to call multiple times — will not create duplicates.
        
        Returns the number of chunks upserted.
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty list.")
            return 0

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            total += len(batch)
            logger.debug("Upserted batch %d–%d (%d total so far)", i, i + len(batch), total)

        logger.info("Upserted %d chunks into collection '%s'.", total, CHROMA_COLLECTION)
        return total

    def clear(self) -> None:
        """Delete all documents from the collection (keeps the collection itself)."""
        count = self._collection.count()
        if count > 0:
            all_ids = self._collection.get(include=[])["ids"]
            self._collection.delete(ids=all_ids)
            logger.info("Cleared %d documents from collection.", count)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K_RESULTS,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most semantically similar chunks to query_text.
        
        Args:
            query_text: The user's question or search string.
            top_k: Number of results to return.
            where_filter: Optional ChromaDB metadata filter, e.g.:
                          {"parcel_id": "A"} to restrict to Parcel A chunks.
        
        Returns:
            List of result dicts, each with keys:
              - 'id': chunk ID
              - 'text': chunk content
              - 'metadata': dict of metadata fields
              - 'distance': L2 distance (lower = more similar)
        """
        query_params: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        raw = self._collection.query(**query_params)

        results: list[dict[str, Any]] = []
        for doc, meta, dist, doc_id in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
            raw["ids"][0],
        ):
            # Filter out chunks that are too dissimilar
            if dist <= MAX_DISTANCE_THRESHOLD:
                results.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                })

        return results
