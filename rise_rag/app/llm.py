"""
llm.py — LLM interface for RISE RAG.

Sends context + question to Ollama (local, free, no API key).
Falls back gracefully if Ollama is not running.

Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT,
    SYSTEM_PROMPT,
    CONTEXT_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Sentinel returned when Ollama is unavailable
OLLAMA_UNAVAILABLE = "__OLLAMA_UNAVAILABLE__"


# ─── Availability Check ───────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Return list of model names available in the local Ollama install."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ─── LLM Client ───────────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Wrapper around the Ollama local inference API.
    
    Uses the /api/generate endpoint with streaming disabled for simplicity,
    or streaming enabled for real-time output.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = OLLAMA_TEMPERATURE,
        num_ctx: int = OLLAMA_NUM_CTX,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._available: bool | None = None  # Cached availability check

    def _check_available(self) -> bool:
        if self._available is None:
            self._available = is_ollama_running()
        return self._available

    def generate(self, context: str, question: str) -> str:
        """
        Generate an answer given retrieved context and the user question.
        
        Returns the full response string, or OLLAMA_UNAVAILABLE if Ollama
        is not running.
        """
        if not self._check_available():
            return OLLAMA_UNAVAILABLE

        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %ds.", OLLAMA_TIMEOUT)
            self._available = None  # Reset so next call retries
            return OLLAMA_UNAVAILABLE

        except requests.exceptions.ConnectionError:
            logger.error("Lost connection to Ollama.")
            self._available = False
            return OLLAMA_UNAVAILABLE

        except Exception as e:
            logger.error("Ollama error: %s", e)
            return OLLAMA_UNAVAILABLE

    def stream_generate(self, context: str, question: str) -> Iterator[str]:
        """
        Stream the LLM response token by token.
        
        Yields string fragments as they arrive.
        Yields OLLAMA_UNAVAILABLE as the only item if Ollama is down.
        """
        if not self._check_available():
            yield OLLAMA_UNAVAILABLE
            return

        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=OLLAMA_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.ConnectionError:
            self._available = False
            yield OLLAMA_UNAVAILABLE

        except Exception as e:
            logger.error("Ollama streaming error: %s", e)
            yield OLLAMA_UNAVAILABLE


# ─── Fallback Response ────────────────────────────────────────────────────────

def build_fallback_response(context: str, question: str) -> str:
    """
    When Ollama is not available, return a helpful response that surfaces
    the retrieved context directly.
    """
    return (
        "  Ollama is not running. To get AI-generated answers, start Ollama:\n"
        "    ollama serve\n"
        "    (and make sure your model is pulled: ollama pull mistral)\n\n"
        "Here is the raw context retrieved from the RISE knowledge base for your question:\n\n"
        "─" * 60 + "\n\n"
        + context
    )
