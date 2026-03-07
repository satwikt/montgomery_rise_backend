"""
chatbot.py — RISE RAG chatbot orchestrator.

This is the main entry point for the chat logic. It:
1. Initialises the VectorStore and LLM
2. Accepts user questions
3. Retrieves relevant context
4. Generates and returns answers

Can be used programmatically or via the CLI (chat.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from .config import TOP_K_RESULTS
from .embeddings import VectorStore
from .llm import OllamaLLM, build_fallback_response, OLLAMA_UNAVAILABLE
from .retriever import Retriever

logger = logging.getLogger(__name__)


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """The full response from the chatbot for a single query."""
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)  # Retrieved chunk metadata
    used_fallback: bool = False                          # True if Ollama was down
    num_chunks_retrieved: int = 0


# ─── Chatbot Class ────────────────────────────────────────────────────────────

class RiseChatbot:
    """
    The RISE RAG chatbot.
    
    Ties together the Retriever and OllamaLLM to answer questions
    about Montgomery's vacant parcel programme.
    
    Example usage:
        bot = RiseChatbot()
        response = bot.ask("What is the heritage score for Parcel A?")
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm: OllamaLLM | None = None,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        """
        Initialise the chatbot.
        
        Args:
            vector_store: A VectorStore instance. If None, creates one.
            llm: An OllamaLLM instance. If None, creates one.
            top_k: Number of chunks to retrieve per query.
        """
        self._store = vector_store or VectorStore()
        self._llm = llm or OllamaLLM()
        self._retriever = Retriever(self._store)
        self._top_k = top_k

        # Simple in-memory conversation history (last N exchanges)
        self._history: list[dict[str, str]] = []
        self._max_history = 6  # Keep last 3 turns (user + assistant each)

    # ─── Public API ───────────────────────────────────────────────────────────

    def ask(self, question: str, parcel_filter: str | None = None) -> ChatResponse:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question.
            parcel_filter: Optionally restrict retrieval to a specific parcel
                           ("A", "B", or "C"). None = search all.
        
        Returns:
            A ChatResponse with the answer and source metadata.
        """
        # Enrich query with recent history for follow-up questions
        enriched_query = self._enrich_query(question)

        # Retrieve relevant context
        context, results = self._retriever.retrieve_and_build_context(
            enriched_query,
            top_k=self._top_k,
            parcel_filter=parcel_filter,
        )

        # Generate answer
        raw_answer = self._llm.generate(context, question)
        used_fallback = raw_answer == OLLAMA_UNAVAILABLE

        if used_fallback:
            answer = build_fallback_response(context, question)
        else:
            answer = raw_answer

        # Update conversation history
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer[:500]})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

        sources = [r["metadata"] for r in results]

        return ChatResponse(
            question=question,
            answer=answer,
            sources=sources,
            used_fallback=used_fallback,
            num_chunks_retrieved=len(results),
        )

    def stream_ask(
        self, question: str, parcel_filter: str | None = None
    ) -> Iterator[str]:
        """
        Stream the answer token by token.
        
        Yields string fragments. The last item will be a special sentinel
        '__DONE__' so callers know streaming is complete.
        
        Note: Sources are not returned in streaming mode. Use ask() if you
        need source attribution.
        """
        enriched_query = self._enrich_query(question)

        context, _ = self._retriever.retrieve_and_build_context(
            enriched_query,
            top_k=self._top_k,
            parcel_filter=parcel_filter,
        )

        full_response: list[str] = []
        for token in self._llm.stream_generate(context, question):
            if token == OLLAMA_UNAVAILABLE:
                yield build_fallback_response(context, question)
                yield "__DONE__"
                return
            full_response.append(token)
            yield token

        # Update history with completed streamed response
        complete = "".join(full_response)
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": complete[:500]})

        yield "__DONE__"

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def is_ready(self) -> bool:
        """Return True if the vector store has documents loaded."""
        return self._store.count() > 0

    def knowledge_base_size(self) -> int:
        """Return the number of chunks in the vector store."""
        return self._store.count()

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _enrich_query(self, question: str) -> str:
        """
        Prepend recent conversation context to improve retrieval for
        follow-up questions like "What about Parcel B?" or "How does that compare?"
        """
        if not self._history:
            return question

        # Look at the last user turn for context
        last_user_turns = [
            h["content"] for h in self._history[-4:] if h["role"] == "user"
        ]
        if not last_user_turns:
            return question

        # Simple enrichment: prepend the last question if this looks like a follow-up
        follow_up_signals = {
            "what about", "how about", "and parcel", "that parcel",
            "the other", "compare", "difference", "same", "also",
            "what does", "what is", "it score", "its score",
        }
        q_lower = question.lower()
        is_follow_up = any(sig in q_lower for sig in follow_up_signals)

        if is_follow_up and last_user_turns:
            return f"{last_user_turns[-1]} {question}"

        return question
