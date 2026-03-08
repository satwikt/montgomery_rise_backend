"""
embeddings.py — ChromaDB vector-store wrapper for the RISE RAG subsystem.

Responsibilities
----------------
* Create or open the persistent ChromaDB collection.
* Embed document chunks with sentence-transformers (local, free, no API key).
* Upsert chunks safely — re-running ingestion never duplicates records.
* Execute semantic similarity queries with optional metadata filtering.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions

from .config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    MAX_DISTANCE_THRESHOLD,
    TOP_K_RESULTS,
)
from .ingestion import Chunk

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding function
# ─────────────────────────────────────────────────────────────────────────────


def _get_embedding_function() -> (
    embedding_functions.SentenceTransformerEmbeddingFunction
):
    """
    Return a ChromaDB-compatible sentence-transformer embedding function.

    Model: ``all-MiniLM-L6-v2``
      - 384-dimensional vectors
      - ~80 MB download, cached after first run
      - Fast inference; strong semantic quality for retrieval tasks
      - Completely free — no API key required
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────────────────────


class VectorStore:
    """
    Thin wrapper around ChromaDB for knowledge-base storage and retrieval.

    The collection is created if it does not yet exist, or reopened from
    ``config.CHROMA_PERSIST_DIR`` if it does.  The store uses L2 (Euclidean)
    distance — lower values mean higher similarity.

    Usage
    -----
    ::

        store = VectorStore()
        store.upsert_chunks(chunks)
        results = store.query("What is the heritage score for Parcel A?")
    """

    def __init__(self) -> None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self._embed_fn = _get_embedding_function()
        self._collection: Collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "l2"},
        )

        logger.info(
            "VectorStore ready — collection '%s' contains %d documents.",
            CHROMA_COLLECTION,
            self._collection.count(),
        )

    # ── write operations ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> int:
        """
        Upsert *chunks* into the ChromaDB collection.

        Upsert semantics mean it is safe to call this multiple times; existing
        records are updated in-place and no duplicates are created.

        Parameters
        ----------
        chunks:
            Parsed document chunks to store.
        batch_size:
            Number of chunks per ChromaDB upsert call.  Smaller batches use
            less memory; larger batches are faster.

        Returns
        -------
        int
            Total number of chunks upserted.
        """
        if not chunks:
            logger.warning("upsert_chunks() called with an empty list — nothing to do.")
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
            logger.debug(
                "Upserted batch [%d:%d] — %d chunks ingested so far.",
                i,
                i + len(batch),
                total,
            )

        logger.info(
            "Upserted %d chunks into collection '%s'.", total, CHROMA_COLLECTION
        )
        return total

    def clear(self) -> None:
        """
        Delete all documents from the collection.

        The collection itself is preserved so subsequent upserts work
        without recreation.
        """
        count = self._collection.count()
        if count == 0:
            logger.info("Collection is already empty — nothing to clear.")
            return
        all_ids: list[str] = self._collection.get(include=[])["ids"]
        self._collection.delete(ids=all_ids)
        logger.info("Cleared %d documents from collection '%s'.", count, CHROMA_COLLECTION)

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        return self._collection.count()

    # ── read operations ───────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K_RESULTS,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the *top_k* most semantically similar chunks to *query_text*.

        Parameters
        ----------
        query_text:
            The user's question or search string.
        top_k:
            Maximum number of results to return.
        where_filter:
            Optional ChromaDB metadata filter, e.g.
            ``{"parcel_id": {"$in": ["A", "general"]}}`` to restrict results
            to Parcel A and general knowledge-base chunks.

        Returns
        -------
        list[dict]
            Each dict contains the keys ``id``, ``text``, ``metadata``, and
            ``distance``.  Results are pre-filtered by
            ``MAX_DISTANCE_THRESHOLD`` so very dissimilar chunks are excluded.
        """
        collection_size = self._collection.count()
        if collection_size == 0:
            logger.warning("query() called on an empty collection.")
            return []

        query_params: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, collection_size),
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
            if dist <= MAX_DISTANCE_THRESHOLD:
                results.append(
                    {"id": doc_id, "text": doc, "metadata": meta, "distance": dist}
                )

        return results
