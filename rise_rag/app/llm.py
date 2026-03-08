"""
llm.py — LLM interface for RISE RAG.

Uses the Groq cloud API to generate answers.
Free tier — no credit card required.

Getting your free Groq API key:
    1. Go to https://console.groq.com
    2. Sign up with your Google or GitHub account
    3. Click "API Keys" in the left sidebar
    4. Click "Create API Key" and copy it
    5. Add it to your .env file: GROQ_API_KEY=your_key_here

Groq free tier limits:
    - 14,400 requests per day
    - 30 requests per minute
    - No credit card required

Docs: https://console.groq.com/docs
"""

from __future__ import annotations

import logging
from typing import Iterator

from groq import Groq, APIConnectionError, APIStatusError, RateLimitError

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
    GROQ_TIMEOUT,
    SYSTEM_PROMPT,
    CONTEXT_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Sentinel returned when Groq is unavailable
LLM_UNAVAILABLE = "__LLM_UNAVAILABLE__"


# ─── Availability Check ───────────────────────────────────────────────────────

def is_llm_available() -> bool:
    """Returns True if a Groq API key is configured."""
    return bool(GROQ_API_KEY.strip())


def get_llm_info() -> dict:
    """Return info about the configured LLM for status endpoints."""
    return {
        "provider": "Groq",
        "model": GROQ_MODEL,
        "configured": is_llm_available(),
        "hosting": "cloud — no local server required",
    }


# ─── LLM Client ───────────────────────────────────────────────────────────────

class GroqLLM:
    """
    Wrapper around the Groq cloud inference API.

    Groq hosts open-source models (Llama, Mixtral) on their own fast
    inference hardware. Free tier with no credit card required.
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

    def _get_client(self) -> Groq | None:
        """Lazily initialise the Groq client. Returns None if key not set."""
        if self._client is not None:
            return self._client
        if not GROQ_API_KEY.strip():
            logger.warning(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com"
            )
            return None
        self._client = Groq(api_key=GROQ_API_KEY, timeout=self.timeout)
        return self._client

    def generate(self, context: str, question: str) -> str:
        """
        Generate an answer given retrieved context and the user question.
        Returns the answer string, or LLM_UNAVAILABLE if the API fails.
        """
        client = self._get_client()
        if client is None:
            return LLM_UNAVAILABLE

        user_message = CONTEXT_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content.strip()

        except RateLimitError:
            logger.warning(
                "Groq rate limit reached. "
                "Free tier allows 30 requests/minute and 14,400/day."
            )
            return LLM_UNAVAILABLE

        except APIConnectionError as e:
            logger.error("Could not connect to Groq API: %s", e)
            return LLM_UNAVAILABLE

        except APIStatusError as e:
            logger.error("Groq API error %s: %s", e.status_code, e.message)
            return LLM_UNAVAILABLE

        except Exception as e:
            logger.error("Unexpected Groq error: %s", repr(e))
            return LLM_UNAVAILABLE

    def stream_generate(self, context: str, question: str) -> Iterator[str]:
        """
        Stream the response token by token.
        Yields LLM_UNAVAILABLE as the only item if the API is unavailable.
        """
        client = self._get_client()
        if client is None:
            yield LLM_UNAVAILABLE
            return

        user_message = CONTEXT_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        try:
            with client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ) as stream:
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token

        except RateLimitError:
            logger.warning("Groq rate limit reached during streaming.")
            yield LLM_UNAVAILABLE

        except Exception as e:
            logger.error("Groq streaming error: %s", repr(e))
            yield LLM_UNAVAILABLE


# ─── Fallback Response ────────────────────────────────────────────────────────

def build_fallback_response(context: str, question: str) -> str:
    """
    When Groq is not available, surface the retrieved context directly
    so the user still gets useful information.
    """
    return (
        "Groq API is not configured or unavailable.\n"
        "To enable AI-generated answers:\n"
        "  1. Get a free key at https://console.groq.com\n"
        "  2. Add it to your .env file: GROQ_API_KEY=your_key_here\n\n"
        "Here is the raw context retrieved from the RISE knowledge base "
        "for your question:\n\n"
        + "-" * 60 + "\n\n"
        + context
    )
