"""
retriever.py — Query logic for RISE RAG.

Takes a user question, retrieves relevant chunks from ChromaDB,
and assembles them into a formatted context string for the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from .embeddings import VectorStore
from .config import TOP_K_RESULTS

logger = logging.getLogger(__name__)


# ─── Retriever Class ──────────────────────────────────────────────────────────

class Retriever:
    """
    Retrieves relevant knowledge base chunks for a given user query.
    
    Handles:
    - Semantic similarity search via VectorStore
    - Optional parcel-scoped filtering
    - Context assembly for prompt injection
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        parcel_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User question string.
            top_k: Max number of chunks to return.
            parcel_filter: If set to "A", "B", or "C", restricts search to
                           chunks for that specific parcel (plus "general" chunks).
                           If None, searches all chunks.
        
        Returns:
            List of result dicts (id, text, metadata, distance).
        """
        where_filter = None
        if parcel_filter and parcel_filter.upper() in {"A", "B", "C"}:
            # ChromaDB $in operator to match either the specific parcel or general
            where_filter = {
                "parcel_id": {"$in": [parcel_filter.upper(), "general"]}
            }

        results = self._store.query(
            query_text=query,
            top_k=top_k,
            where_filter=where_filter,
        )

        logger.debug(
            "Retrieved %d chunks for query: '%s'", len(results), query[:80]
        )
        return results

    def build_context(self, results: list[dict[str, Any]]) -> str:
        """
        Assemble retrieved chunks into a single context string for the LLM.
        
        Each chunk is separated by a clear divider and prefixed with its
        source metadata so the LLM can reference it if needed.
        """
        if not results:
            return "No relevant context found in the knowledge base."

        parts: list[str] = []
        for i, result in enumerate(results, start=1):
            meta = result["metadata"]
            source = meta.get("source_file", "unknown")
            parcel = meta.get("parcel_id", "general")
            topic = meta.get("topic", "general")

            header = f"[Source {i}: {source} | parcel={parcel} | topic={topic}]"
            parts.append(f"{header}\n{result['text']}")

        return "\n\n---\n\n".join(parts)

    def retrieve_and_build_context(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        parcel_filter: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Convenience method: retrieve chunks and return both context string
        and the raw results list.
        
        Returns:
            (context_string, results_list)
        """
        results = self.retrieve(query, top_k=top_k, parcel_filter=parcel_filter)
        context = self.build_context(results)
        return context, results
