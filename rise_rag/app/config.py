"""
config.py — Centralised configuration for the RISE RAG subsystem.

All tuneable parameters live here.  Import constants from this module;
never hard-code values in other modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# rise_rag/app/config.py  →  rise_rag/app/  →  rise_rag/  →  project root
_APP_DIR: Path = Path(__file__).resolve().parent
RAG_ROOT: Path = _APP_DIR.parent           # rise_rag/
PROJECT_ROOT: Path = RAG_ROOT.parent       # repo root (where api.py lives)

DATA_DIR: Path = RAG_ROOT / "data"
CHROMA_PERSIST_DIR: Path = RAG_ROOT / "chroma_db"
CHROMA_COLLECTION: str = "rise_knowledge"

# ─────────────────────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────────────────────

# Free, runs locally, no API key required.
# ~80 MB download from HuggingFace, cached after first run.
# 384-dimensional vectors; excellent semantic quality for retrieval tasks.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Groq LLM
# ─────────────────────────────────────────────────────────────────────────────

# Free tier — 14,400 req/day, 30 req/min, no credit card required.
# Get your key at: https://console.groq.com
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# Available free-tier models (best → fastest):
#   llama-3.3-70b-versatile   best quality, recommended
#   llama-3.1-8b-instant      faster, still very capable
#   mixtral-8x7b-32768        large context window
GROQ_MODEL: str = "llama-3.3-70b-versatile"

GROQ_TEMPERATURE: float = 0.1   # Low = factual, grounded — ideal for RAG
GROQ_MAX_TOKENS: int = 1024
GROQ_TIMEOUT: int = 30          # seconds

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_RESULTS: int = 5
MAX_DISTANCE_THRESHOLD: float = 1.8   # L2 distance; higher = less similar

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENT_SEPARATOR: str = "---"
MAX_CHUNK_SIZE: int = 2000   # characters

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are the RISE chatbot — the AI assistant for the \
RISE (Revitalization Intelligence and Smart Empowerment) system built for the \
City of Montgomery, Alabama.

RISE helps city planners identify the highest-impact reuse options for \
city-owned vacant parcels using real data: foot traffic, flood risk, 311 \
distress signals, civil rights heritage proximity, and active federal grant \
windows.

Your job is to answer questions accurately and helpfully using ONLY the \
context provided below.

Rules:
- Answer only from the provided context. Do not invent facts.
- If the answer is not in the context, say: "I don't have that information in \
my knowledge base. I can answer questions about the three RISE hero parcels, \
the scoring model, data sources, and active grants."
- Be specific: cite parcel names, scores, distances, and dollar figures when \
they appear in the context.
- Keep answers concise and clear — this is a planning tool, not a research \
paper.
- When grant deadlines are mentioned, always state the urgency. As of \
March 8 2026: USDA RED Q3 closes in 23 days (March 31) and USDA VAPG closes \
in 38 days (April 15).
- Never provide legal or financial advice. Refer users to USDA Rural \
Development Alabama for grant applications.
"""

CONTEXT_PROMPT_TEMPLATE: str = """\
CONTEXT FROM RISE KNOWLEDGE BASE:
{context}

---
QUESTION: {question}

ANSWER:"""
