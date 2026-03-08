"""
llm.py — Groq cloud LLM interface for the RISE RAG subsystem.

Uses the Groq API (free tier) to generate grounded answers from
retrieved context.  No local GPU or server required.

Free-tier limits (as of 2026)
------------------------------
  14 400 requests / day
  30 requests / minute
  No credit card required

Obtaining an API key
--------------------
  1. Visit https://console.groq.com and sign up with Google or GitHub.
  2. Click "API Keys" → "Create API Key".
  3. Add to your .env file:  GROQ_API_KEY=gsk_…
"""

from __future__ import annotations

import logging
from typing import Iterator

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from .config import (
    CONTEXT_PROMPT_TEMPLATE,
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_TIMEOUT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Sentinel value returned / yielded when the LLM is unavailable so callers
# can distinguish a genuine empty answer from an infrastructure failure.
LLM_UNAVAILABLE: str = "__LLM_UNAVAILABLE__"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def is_llm_available() -> bool:
    """Return ``True`` if a Groq API key is present in the environment."""
    return bool(GROQ_API_KEY.strip())


def get_llm_info() -> dict[str, str | bool]:
    """Return a status dict suitable for health-check endpoints."""
    return {
        "provider": "Groq",
        "model": GROQ_MODEL,
        "configured": is_llm_available(),
        "hosting": "cloud — no local server required",
    }


def build_fallback_response(context: str, question: str) -> str:  # noqa: ARG001
    """
    Return a human-readable fallback when Groq is unavailable.

    Surfaces the raw retrieved context so the user still receives
    useful information even without an LLM-generated answer.
    """
    return (
        "Groq API is not configured or unavailable.\n"
        "To enable AI-generated answers:\n"
        "  1. Get a free key at https://console.groq.com\n"
        "  2. Set the environment variable:  GROQ_API_KEY=gsk_...\n\n"
        "Raw context retrieved from the RISE knowledge base:\n"
        + "-" * 60
        + "\n\n"
        + context
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────


class GroqLLM:
    """
    Thin, stateful wrapper around the Groq chat-completions API.

    The underlying ``Groq`` client is created lazily on the first call so
    that importing this module never raises even when the key is absent.

    Parameters
    ----------
    model:
        Groq model identifier.  Defaults to ``config.GROQ_MODEL``.
    temperature:
        Sampling temperature (0 = fully deterministic).
    max_tokens:
        Maximum tokens in the generated response.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        temperature: float = GROQ_TEMPERATURE,
        max_tokens: int = GROQ_MAX_TOKENS,
        timeout: int = GROQ_TIMEOUT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Groq | None = None

    # ── private ──────────────────────────────────────────────────────────────

    def _get_client(self) -> Groq | None:
        """Lazily initialise the Groq client.  Returns ``None`` if no key."""
        if self._client is not None:
            return self._client
        if not GROQ_API_KEY.strip():
            logger.warning(
                "GROQ_API_KEY is not set — get a free key at https://console.groq.com"
            )
            return None
        self._client = Groq(api_key=GROQ_API_KEY, timeout=self.timeout)
        return self._client

    def _build_messages(self, context: str, question: str) -> list[dict[str, str]]:
        """Construct the chat message list for a RAG completion request."""
        user_content = CONTEXT_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ── public ───────────────────────────────────────────────────────────────

    def generate(self, context: str, question: str) -> str:
        """
        Generate a grounded answer from retrieved *context* and *question*.

        Returns
        -------
        str
            The assistant's answer, or ``LLM_UNAVAILABLE`` if the API
            call fails for any reason.
        """
        client = self._get_client()
        if client is None:
            return LLM_UNAVAILABLE

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(context, question),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content.strip()

        except RateLimitError:
            logger.warning(
                "Groq rate limit hit — free tier: 30 req/min, 14 400 req/day."
            )
        except APIConnectionError as exc:
            logger.error("Groq connection error: %s", exc)
        except APIStatusError as exc:
            logger.error("Groq API error %s: %s", exc.status_code, exc.message)
        except Exception:
            logger.exception("Unexpected error in GroqLLM.generate()")

        return LLM_UNAVAILABLE

    def stream_generate(self, context: str, question: str) -> Iterator[str]:
        """
        Stream the response token-by-token.

        Yields
        ------
        str
            Individual tokens.  If the API is unavailable, yields a single
            ``LLM_UNAVAILABLE`` token and then the generator closes.
        """
        client = self._get_client()
        if client is None:
            yield LLM_UNAVAILABLE
            return

        try:
            with client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(context, question),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ) as stream:
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token

        except RateLimitError:
            logger.warning("Groq rate limit hit during streaming.")
            yield LLM_UNAVAILABLE
        except Exception:
            logger.exception("Unexpected error in GroqLLM.stream_generate()")
            yield LLM_UNAVAILABLE
