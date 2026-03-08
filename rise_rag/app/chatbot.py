"""
chatbot.py — RAG chatbot orchestrator for the RISE subsystem.

Ties together the :class:`~retriever.Retriever` and :class:`~llm.GroqLLM`
to answer natural-language questions about Montgomery's vacant-parcel
programme, grounded in the RISE knowledge base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from .config import TOP_K_RESULTS
from .embeddings import VectorStore
from .llm import LLM_UNAVAILABLE, GroqLLM, build_fallback_response
from .retriever import Retriever

logger = logging.getLogger(__name__)

# Signals that indicate a follow-up question referencing prior context.
_FOLLOW_UP_SIGNALS: frozenset[str] = frozenset(
    {
        "what about",
        "how about",
        "and parcel",
        "that parcel",
        "the other",
        "compare",
        "difference",
        "same",
        "also",
        "what does",
        "what is",
        "it score",
        "its score",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChatResponse:
    """The full response from the chatbot for a single user query."""

    question: str
    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)
    used_fallback: bool = False
    num_chunks_retrieved: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────────────────────────────────────


class RiseChatbot:
    """
    RAG chatbot for the RISE vacant-parcel programme.

    Combines semantic retrieval (ChromaDB) with Groq LLM generation to
    provide grounded, citation-aware answers.  A short rolling conversation
    history enables coherent follow-up questions.

    Parameters
    ----------
    vector_store:
        Optional pre-built :class:`~embeddings.VectorStore`.  A new one is
        created if not supplied.
    llm:
        Optional pre-configured :class:`~llm.GroqLLM`.  A new one with
        default settings is created if not supplied.
    top_k:
        Number of knowledge-base chunks to retrieve per query.

    Example
    -------
    ::

        bot = RiseChatbot()
        response = bot.ask("What is the heritage score for Parcel A?")
        print(response.answer)
    """

    # Maximum number of conversation turns to keep in memory.
    # Each turn = one user message + one assistant message (2 list items).
    _MAX_HISTORY_TURNS: int = 6

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

    # ── public ────────────────────────────────────────────────────────────────

    def ask(self, question: str, parcel_filter: str | None = None) -> ChatResponse:
        """
        Answer *question* using retrieval-augmented generation.

        Parameters
        ----------
        question:
            The user's natural-language question.
        parcel_filter:
            Optional single-character parcel scope (``'A'``, ``'B'``, or
            ``'C'``).  When set, retrieval is restricted to chunks for that
            parcel plus general knowledge-base chunks.

        Returns
        -------
        ChatResponse
            Contains the answer, source metadata, fallback flag, and
            retrieved-chunk count.
        """
        enriched = self._enrich_query(question)

        context, results = self._retriever.retrieve_and_build_context(
            enriched,
            top_k=self._top_k,
            parcel_filter=parcel_filter,
        )

        raw_answer = self._llm.generate(context, question)
        used_fallback = raw_answer == LLM_UNAVAILABLE
        answer = build_fallback_response(context, question) if used_fallback else raw_answer

        self._record_turn(question, answer)

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
        """
        Stream the answer to *question* token-by-token.

        Yields individual string tokens.  The final token yielded is always
        the sentinel ``'__DONE__'`` so callers can detect end-of-stream.

        If the LLM is unavailable, the full fallback message is yielded as
        a single token followed by ``'__DONE__'``.
        """
        enriched = self._enrich_query(question)
        context, _ = self._retriever.retrieve_and_build_context(
            enriched,
            top_k=self._top_k,
            parcel_filter=parcel_filter,
        )

        full_tokens: list[str] = []

        for token in self._llm.stream_generate(context, question):
            if token == LLM_UNAVAILABLE:
                yield build_fallback_response(context, question)
                yield "__DONE__"
                return
            full_tokens.append(token)
            yield token

        complete_answer = "".join(full_tokens)
        self._record_turn(question, complete_answer)
        yield "__DONE__"

    def clear_history(self) -> None:
        """Wipe the conversation history."""
        self._history.clear()

    def is_ready(self) -> bool:
        """Return ``True`` if the knowledge base contains at least one chunk."""
        return self._store.count() > 0

    def knowledge_base_size(self) -> int:
        """Return the total number of chunks in the knowledge base."""
        return self._store.count()

    # ── private ───────────────────────────────────────────────────────────────

    def _record_turn(self, question: str, answer: str) -> None:
        """
        Append a question/answer pair to the conversation history and trim
        the history to the last ``_MAX_HISTORY_TURNS`` turns.
        """
        self._history.append({"role": "user", "content": question})
        # Store a truncated version of the answer to keep memory usage bounded.
        self._history.append({"role": "assistant", "content": answer[:500]})

        max_items = self._MAX_HISTORY_TURNS * 2
        if len(self._history) > max_items:
            self._history = self._history[-max_items:]

    def _enrich_query(self, question: str) -> str:
        """
        Prepend recent conversation context to *question* when it appears
        to be a follow-up question.

        This allows the retriever to find relevant chunks even when the
        question itself is vague (e.g. "what about Parcel B?").

        Returns
        -------
        str
            The enriched query string, or *question* unchanged if no
            follow-up signals are detected or there is no prior history.
        """
        if not self._history:
            return question

        prior_user_turns = [
            h["content"] for h in self._history[-4:] if h["role"] == "user"
        ]
        if not prior_user_turns:
            return question

        question_lower = question.lower()
        if any(signal in question_lower for signal in _FOLLOW_UP_SIGNALS):
            return f"{prior_user_turns[-1]} {question}"

        return question
