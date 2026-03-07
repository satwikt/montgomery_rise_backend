"""
config.py — All RISE RAG configuration constants.
Edit this file to change models, paths, and tuning parameters.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    class Config:
        extra = "ignore"   # ← add this line
        env_file = ".env"
# ─── Paths ────────────────────────────────────────────────────────────────────

# Root of the project (parent of the rise_rag package dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the .txt knowledge base files live
DATA_DIR = PROJECT_ROOT / "data"

# Where ChromaDB persists its vector database
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# ChromaDB collection name
CHROMA_COLLECTION = "rise_knowledge"

# ─── Embedding Model ──────────────────────────────────────────────────────────

# Free, local, no API key. Downloaded once from HuggingFace and cached.
# all-MiniLM-L6-v2 is fast and produces 384-dim vectors — ideal for this scale.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Ollama LLM ───────────────────────────────────────────────────────────────

# Ollama server URL (default local install)
OLLAMA_BASE_URL = "http://localhost:11434"

# Model to use — options (in order of quality vs speed):
#   mistral        ~4GB — best quality, recommended
#   llama3.2       ~2GB — fast, good quality
#   phi3           ~2GB — very fast, lightweight
#   gemma2:2b      ~1.6GB — smallest option
OLLAMA_MODEL = "mistral"

# Ollama generation parameters
OLLAMA_TEMPERATURE = 0.1      # Low = more factual, less creative. Good for RAG.
OLLAMA_NUM_CTX = 4096         # Context window size (tokens)
OLLAMA_TIMEOUT = 120          # Seconds to wait for a response

# ─── Retrieval ────────────────────────────────────────────────────────────────

# Number of chunks to retrieve per query
TOP_K_RESULTS = 5

# Minimum similarity distance threshold (ChromaDB uses L2 distance by default;
# lower = more similar. Chunks with distance > this are considered irrelevant.)
MAX_DISTANCE_THRESHOLD = 1.8

# ─── Chunking ─────────────────────────────────────────────────────────────────

# Document separator used in the .txt knowledge base files
DOCUMENT_SEPARATOR = "---"

# Maximum characters per chunk (for any chunks that don't have natural separators)
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
