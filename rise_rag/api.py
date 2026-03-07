"""
api.py — RISE RAG FastAPI Server
==================================
Wraps the RISE RAG chatbot and exposes 4 endpoints that mirror
the structure of the project's main RISE API:

  GET  /                          Health check + system status
  GET  /parcels                   List all 3 hero parcels (metadata only)
  GET  /parcels/{id}/answer       Full AI summary for a hero parcel (A | B | C)
  POST /parcels/ask               Ask a free-form question across all parcels

No paid APIs. No Gemini. No API keys required.
Powered by ChromaDB + sentence-transformers + Ollama (local, free).

Setup:
    pip install -r requirements.txt

Run:
    uvicorn api:app --reload --port 8000

Then open:
    http://localhost:8000/docs    (interactive API docs)
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chatbot import RiseChatbot
from app.llm import is_ollama_running, get_available_models
from app.config import OLLAMA_MODEL

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ─── Hero Parcel Metadata ─────────────────────────────────────────────────────
# Static metadata for the three hero parcels.
# Matches the parcel data defined in the RISE knowledge base.

HERO_PARCELS = {
    "A": {
        "id": "A",
        "label": "Parcel A — Heritage",
        "story": "Heritage Story",
        "address": "Commerce Street, Montgomery, Alabama 36104",
        "parcel_id": "11 01 12 4 004 001.000",
        "acres": 8.34,
        "zone_context": "heritage",
        "nearest_anchor": "Rosa Parks Museum",
        "min_dist_miles": 0.154,
        "coords": {"lat": 32.3789, "lon": -86.3109},
        "rise_score": 74,
        "urgency": "medium",
        "top_recommendation": "Civil Rights Heritage Plaza",
        "open_grants": 2,
    },
    "B": {
        "id": "B",
        "label": "Parcel B — IX Hub",
        "story": "Smart Infrastructure Story",
        "address": "643 Kimball Street, Montgomery, Alabama 36108",
        "parcel_id": "11 05 15 1 010 022.000",
        "acres": 0.33,
        "zone_context": "ix_hub",
        "nearest_anchor": "Maxwell AFB Gate",
        "min_dist_miles": 0.891,
        "coords": {"lat": 32.3684, "lon": -86.3439},
        "rise_score": 29,
        "urgency": "low",
        "top_recommendation": "AI Workforce Training Hub",
        "open_grants": 1,
    },
    "C": {
        "id": "C",
        "label": "Parcel C — Food Desert",
        "story": "Economic Urgency",
        "address": "Coosa Street, Montgomery, Alabama 36104",
        "parcel_id": "11 01 12 4 004 001.000",
        "acres": 8.34,
        "zone_context": "food_desert",
        "nearest_anchor": "MGMix Internet Exchange",
        "min_dist_miles": 0.117,
        "coords": {"lat": 32.3792, "lon": -86.3087},
        "rise_score": 72,
        "urgency": "high",
        "top_recommendation": "Community Grocery Co-op",
        "open_grants": 2,
    },
}

# Pre-built questions sent to the RAG chatbot for each parcel's /answer endpoint.
# Each question asks for a comprehensive summary so the response is useful out of the box.
PARCEL_SUMMARY_QUESTIONS = {
    "A": (
        "Give a full summary of Parcel A including its RISE score across all 8 dimensions, "
        "its top 3 recommendations, available grants, and community health context."
    ),
    "B": (
        "Give a full summary of Parcel B including its RISE score across all 8 dimensions, "
        "its top 3 recommendations, available grants, and workforce context."
    ),
    "C": (
        "Give a full summary of Parcel C including its RISE score across all 8 dimensions, "
        "its top 3 recommendations, available grants, and food desert health context."
    ),
}

# ─── Global Chatbot Instance ──────────────────────────────────────────────────
# Initialised once at server startup. Reused across all requests.
# Avoids reloading ChromaDB and the embedding model on every call.

chatbot: RiseChatbot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    global chatbot
    logger.info("Starting RISE RAG API — initialising chatbot...")
    try:
        chatbot = RiseChatbot()
        if not chatbot.is_ready():
            logger.warning(
                "Knowledge base is empty. Run: python scripts/ingest.py"
            )
        else:
            logger.info(
                "Chatbot ready. Knowledge base: %d chunks.",
                chatbot.knowledge_base_size(),
            )
    except Exception as e:
        logger.error("Failed to initialise chatbot: %s", e)
        chatbot = None

    yield  # Server is live here

    logger.info("RISE RAG API shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Montgomery RISE RAG API",
    description=(
        "Revitalization Intelligence and Smart Empowerment — "
        "RAG-powered chatbot API for Montgomery, Alabama's vacant parcel programme. "
        "Powered by ChromaDB, sentence-transformers, and Ollama. No paid APIs required."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow any frontend or tool (Postman, browser, colleague's app) to call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class AskRequest(BaseModel):
    """Body for POST /parcels/ask"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Your question about RISE parcels, scoring, grants, or community context.",
        examples=["What is the most urgent grant right now?"],
    )
    parcel_filter: Optional[str] = Field(
        default=None,
        description=(
            "Restrict the search to one parcel. Must be 'A', 'B', or 'C'. "
            "Leave blank to search across all knowledge."
        ),
        examples=["C"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What grants are available for the food desert parcel?",
                    "parcel_filter": "C",
                },
                {
                    "question": "What is RISE and how does the scoring work?",
                    "parcel_filter": None,
                },
            ]
        }
    }


class SourceModel(BaseModel):
    document_title: str
    parcel_id: str
    topic: str
    source_file: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    num_chunks_retrieved: int
    used_fallback: bool
    parcel_filter: Optional[str]
    generated_at: str


class ParcelSummaryResponse(BaseModel):
    parcel_id: str
    label: str
    story: str
    address: str
    acres: float
    zone_context: str
    nearest_anchor: str
    min_dist_miles: float
    rise_score: int
    urgency: str
    top_recommendation: str
    open_grants: int
    summary: str
    sources: list[SourceModel]
    num_chunks_retrieved: int
    used_fallback: bool
    generated_at: str


# ─── Shared Helpers ───────────────────────────────────────────────────────────

def _check_chatbot_ready():
    """Raise HTTP 503 if the chatbot is not initialised or knowledge base is empty."""
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialised. Check server logs.",
        )
    if not chatbot.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Run: python scripts/ingest.py",
        )


def _build_sources(raw_sources: list[dict]) -> list[SourceModel]:
    """Convert raw metadata dicts from the chatbot into SourceModel objects."""
    return [
        SourceModel(
            document_title=s.get("document_title", "Unknown"),
            parcel_id=s.get("parcel_id", "general"),
            topic=s.get("topic", "general"),
            source_file=s.get("source_file", "unknown"),
        )
        for s in raw_sources
    ]


# ─── Endpoint 1 — Health Check ────────────────────────────────────────────────

@app.get("/", summary="Health check", tags=["System"])
def health_check():
    """
    Returns API status, knowledge base size, and Ollama availability.
    Mirrors the health check endpoint in the main RISE API.
    """
    ollama_ok = is_ollama_running()
    kb_size = chatbot.knowledge_base_size() if chatbot else 0
    kb_ready = chatbot.is_ready() if chatbot else False

    return {
        "status": "ok" if kb_ready else "degraded",
        "service": "Montgomery RISE RAG API",
        "version": "1.0.0",
        "knowledge_base_chunks": kb_size,
        "knowledge_base_ready": kb_ready,
        "ollama_running": ollama_ok,
        "ollama_model": OLLAMA_MODEL,
        "llm_source": "ollama-local" if ollama_ok else "fallback-context-only",
        "parcels": len(HERO_PARCELS),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─── Endpoint 2 — List Hero Parcels ──────────────────────────────────────────

@app.get("/parcels", summary="List hero parcels", tags=["Parcels"])
def list_parcels():
    """
    Returns lightweight metadata for all 3 hero parcels.
    No RAG call is made here — this is static reference data.
    Use GET /parcels/{id}/answer to get a full AI-generated summary for a parcel.
    """
    return {
        "parcels": list(HERO_PARCELS.values()),
        "total": len(HERO_PARCELS),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ─── Endpoint 3 — Full AI Summary for a Hero Parcel ──────────────────────────

@app.get(
    "/parcels/{parcel_id}/answer",
    response_model=ParcelSummaryResponse,
    summary="Get full AI summary for a hero parcel",
    tags=["Parcels"],
)
def answer_parcel(
    parcel_id: str = Path(
        ...,
        description="Hero parcel identifier: A, B, or C",
        example="A",
    )
):
    """
    Returns a full AI-generated summary for one of the 3 hero parcels including:

    - All 8 RISE dimension scores
    - Top 3 land reuse recommendations
    - Available grant windows and deadlines
    - Community health and workforce context

    Powered by the RAG chatbot — retrieves the most relevant knowledge base
    chunks for the parcel then generates a plain-English summary using Ollama.
    """
    pid = parcel_id.upper().strip()
    if pid not in HERO_PARCELS:
        raise HTTPException(
            status_code=404,
            detail=f"Parcel '{parcel_id}' not found. Valid IDs are: A, B, C",
        )

    _check_chatbot_ready()

    parcel_meta = HERO_PARCELS[pid]
    question = PARCEL_SUMMARY_QUESTIONS[pid]

    try:
        response = chatbot.ask(question, parcel_filter=pid)
    except Exception as e:
        logger.error("Error generating parcel summary for %s: %s", pid, e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate parcel summary.",
        )

    return ParcelSummaryResponse(
        parcel_id=pid,
        label=parcel_meta["label"],
        story=parcel_meta["story"],
        address=parcel_meta["address"],
        acres=parcel_meta["acres"],
        zone_context=parcel_meta["zone_context"],
        nearest_anchor=parcel_meta["nearest_anchor"],
        min_dist_miles=parcel_meta["min_dist_miles"],
        rise_score=parcel_meta["rise_score"],
        urgency=parcel_meta["urgency"],
        top_recommendation=parcel_meta["top_recommendation"],
        open_grants=parcel_meta["open_grants"],
        summary=response.answer,
        sources=_build_sources(response.sources),
        num_chunks_retrieved=response.num_chunks_retrieved,
        used_fallback=response.used_fallback,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


# ─── Endpoint 4 — Free-Form Question ─────────────────────────────────────────

@app.post(
    "/parcels/ask",
    response_model=AskResponse,
    summary="Ask a free-form question",
    tags=["Parcels"],
)
def ask_question(body: AskRequest):
    """
    Ask any question about the RISE system, parcels, scoring, or grants.

    Optionally set **parcel_filter** to restrict the knowledge search to a
    specific parcel — useful for targeted questions about one parcel.

    Example questions:
    - "What is the most urgent grant right now?"
    - "How does the 311 distress score work?"
    - "How much would the city pay for a grocery store?" (parcel_filter: C)
    - "What is the heritage boost and why does it matter?" (parcel_filter: A)
    """
    _check_chatbot_ready()

    # Validate parcel filter
    parcel_filter = None
    if body.parcel_filter:
        pf = body.parcel_filter.upper().strip()
        if pf not in {"A", "B", "C"}:
            raise HTTPException(
                status_code=422,
                detail="parcel_filter must be 'A', 'B', or 'C'.",
            )
        parcel_filter = pf

    try:
        response = chatbot.ask(body.question, parcel_filter=parcel_filter)
    except Exception as e:
        logger.error("Error generating response: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate a response.",
        )

    return AskResponse(
        question=response.question,
        answer=response.answer,
        sources=_build_sources(response.sources),
        num_chunks_retrieved=response.num_chunks_retrieved,
        used_fallback=response.used_fallback,
        parcel_filter=parcel_filter,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )
