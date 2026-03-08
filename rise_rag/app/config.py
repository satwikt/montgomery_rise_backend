"""
config.py — All RISE RAG configuration constants.
Edit this file to change models, paths, and tuning parameters.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load GROQ_API_KEY from .env file if present
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION = "rise_knowledge"

# ─── Embedding Model ──────────────────────────────────────────────────────────

# Free, local, no API key. Downloaded once from HuggingFace and cached.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Groq LLM ─────────────────────────────────────────────────────────────────

# API key — loaded from environment variable or .env file.
# Get your free key at: https://console.groq.com
# Add to your .env file: GROQ_API_KEY=your_key_here
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Model to use.
# llama-3.3-70b-versatile  — best quality, free tier, recommended
# llama-3.1-8b-instant     — faster, lighter, still very capable
# mixtral-8x7b-32768       — large context window
GROQ_MODEL = "llama-3.3-70b-versatile"

# Generation parameters
GROQ_TEMPERATURE = 0.1    # Low = more factual, less creative. Good for RAG.
GROQ_MAX_TOKENS = 1024    # Maximum tokens in the response.
GROQ_TIMEOUT = 30         # Seconds to wait for a response.

# ─── Retrieval ────────────────────────────────────────────────────────────────

TOP_K_RESULTS = 5
MAX_DISTANCE_THRESHOLD = 1.8

# ─── Chunking ─────────────────────────────────────────────────────────────────

DOCUMENT_SEPARATOR = "---"
MAX_CHUNK_SIZE = 2000

# ─── Prompting ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the RISE chatbot — the AI assistant for the RISE (Revitalization Intelligence and Smart Empowerment) system built for the City of Montgomery, Alabama.

RISE helps city planners identify the highest-impact reuse options for city-owned vacant parcels using real data: foot traffic, flood risk, 311 distress signals, civil rights heritage proximity, and active federal grant windows.

Your job is to answer questions accurately and helpfully using ONLY the context provided below.

Rules:
- Answer only from the provided context. Do not invent facts.
- If the answer is not in the context, say: "I don't have that information in my knowledge base. I can answer questions about the three RISE hero parcels, the scoring model, data sources, and active grants."
- Be specific: cite parcel names, scores, distances, and dollar figures when they appear in the context.
- Keep answers concise and clear — this is a planning tool, not a research paper.
- If a question involves grant deadlines, always mention the urgency.
- Never provide legal or financial advice. Refer users to USDA Rural Development Alabama for grant applications.
"""

CONTEXT_PROMPT_TEMPLATE = """CONTEXT FROM RISE KNOWLEDGE BASE:
{context}

---
QUESTION: {question}

ANSWER:"""
