"""
chatbot.py — RISE RAG chatbot orchestrator.

Ties together the Retriever and GroqLLM to answer questions
about Montgomery's vacant parcel programme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from .config import TOP_K_RESULTS
from .embeddings import VectorStore
from .llm import GroqLLM, build_fallback_response, LLM_UNAVAILABLE
from .retriever import Retriever

logger = logging.getLogger(__name__)


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """The full response from the chatbot for a single query."""
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    used_fallback: bool = False
    num_chunks_retrieved: int = 0


# ─── Chatbot Class ────────────────────────────────────────────────────────────

class RiseChatbot:
    """
    The RISE RAG chatbot.

    Example usage:
        bot = RiseChatbot()
        response = bot.ask("What is the heritage score for Parcel A?")
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm: GroqLLM | None = None,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        self._store = vector_store or VectorStore()
        self._llm = llm or GroqLLM()
        self._retriever = Retriever(self._store)
        self._top_k = top_k
        self._history: list[dict[str, str]] = []
        self._max_history = 6

    def ask(self, question: str, parcel_filter: str | None = None) -> ChatResponse:
        """Answer a question using RAG."""
        enriched_query = self._enrich_query(question)

        context, results = self._retriever.retrieve_and_build_context(
            enriched_query,
            top_k=self._top_k,
            parcel_filter=parcel_filter,
        )

        raw_answer = self._llm.generate(context, question)
        used_fallback = raw_answer == LLM_UNAVAILABLE

        answer = build_fallback_response(context, question) if used_fallback else raw_answer

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer[:500]})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

        return ChatResponse(
            question=question,
            answer=answer,
            sources=[r["metadata"] for r in results],
            used_fallback=used_fallback,
            num_chunks_retrieved=len(results),
        )

    def stream_ask(
        self, question: str, parcel_filter: str | None = None
    ) -> Iterator[str]:
        """Stream the answer token by token. Yields '__DONE__' when complete."""
        enriched_query = self._enrich_query(question)
        context, _ = self._retriever.retrieve_and_build_context(
            enriched_query, top_k=self._top_k, parcel_filter=parcel_filter,
        )

        full_response: list[str] = []
        for token in self._llm.stream_generate(context, question):
            if token == LLM_UNAVAILABLE:
                yield build_fallback_response(context, question)
                yield "__DONE__"
                return
            full_response.append(token)
            yield token

        complete = "".join(full_response)
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": complete[:500]})
        yield "__DONE__"

    def clear_history(self) -> None:
        self._history.clear()

    def is_ready(self) -> bool:
        return self._store.count() > 0

    def knowledge_base_size(self) -> int:
        return self._store.count()

    def _enrich_query(self, question: str) -> str:
        if not self._history:
            return question
        last_user_turns = [h["content"] for h in self._history[-4:] if h["role"] == "user"]
        if not last_user_turns:
            return question
        follow_up_signals = {
            "what about", "how about", "and parcel", "that parcel",
            "the other", "compare", "difference", "same", "also",
            "what does", "what is", "it score", "its score",
        }
        if any(sig in question.lower() for sig in follow_up_signals):
            return f"{last_user_turns[-1]} {question}"
        return question
