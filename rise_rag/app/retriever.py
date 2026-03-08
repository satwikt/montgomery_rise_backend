"""
retriever.py — Semantic retrieval and context assembly for the RISE RAG subsystem.

Responsibilities
----------------
* Accept a natural-language query and optional parcel scope.
* Retrieve the most relevant knowledge-base chunks from ChromaDB.
* Assemble the retrieved chunks into a formatted context string for the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import TOP_K_RESULTS
from .embeddings import VectorStore

logger = logging.getLogger(__name__)

# Parcel IDs that may be used as filter scopes.
_VALID_PARCEL_FILTERS: frozenset[str] = frozenset({"A", "B", "C"})


class Retriever:
    """
    Retrieves relevant knowledge-base chunks for a given user query.

    Supports optional parcel-scoped filtering via ChromaDB metadata
    predicates so that questions about a specific parcel retrieve chunks
    tagged with that parcel (or marked as general knowledge).

    Parameters
    ----------
    vector_store:
        An initialised :class:`~embeddings.VectorStore` instance.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    # ── public ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        parcel_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically relevant chunks for *query*.

        Parameters
        ----------
        query:
            The user's question or enriched search string.
        top_k:
            Maximum number of chunks to return.
        parcel_filter:
            If set to ``'A'``, ``'B'``, or ``'C'``, restricts results to
            chunks tagged with that parcel ID **or** ``'general'``.  ``None``
            searches the entire knowledge base.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``text``, ``metadata``, and
            ``distance`` keys.
        """
        where_filter: dict[str, Any] | None = None

        if parcel_filter is not None:
            pid = parcel_filter.upper()
            if pid in _VALID_PARCEL_FILTERS:
                # ChromaDB ``$in`` operator: match the specific parcel OR general chunks.
                where_filter = {"parcel_id": {"$in": [pid, "general"]}}
            else:
                logger.warning(
                    "Invalid parcel_filter '%s' — ignoring filter.", parcel_filter
                )

        results = self._store.query(
            query_text=query,
            top_k=top_k,
            where_filter=where_filter,
        )

        logger.debug(
            "Retrieved %d chunks for query: '%.80s' (parcel_filter=%s)",
            len(results),
            query,
            parcel_filter,
        )
        return results

    def build_context(self, results: list[dict[str, Any]]) -> str:
        """
        Assemble retrieved chunks into a single context string for the LLM.

        Each chunk is separated by a divider and prefixed with its source
        metadata so the model can attribute information if needed.

        Returns
        -------
        str
            Formatted context string, or a brief message when no chunks
            are available.
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
        Convenience method: retrieve chunks and return both the assembled
        context string and the raw results list.

        Returns
        -------
        tuple[str, list[dict]]
            ``(context_string, results_list)``
        """
        results = self.retrieve(query, top_k=top_k, parcel_filter=parcel_filter)
        context = self.build_context(results)
        return context, results
