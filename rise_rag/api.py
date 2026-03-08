"""
api.py — RISE RAG FastAPI Server

  GET  /                          Health check + system status
  GET  /parcels                   List all 3 hero parcels (metadata only)
  GET  /parcels/{id}/answer       Full AI summary for a hero parcel (A | B | C)
  POST /parcels/ask               Ask a free-form question across all parcels

Powered by ChromaDB + sentence-transformers + Groq (free tier).
No local server or device hosting required.

Setup:
    pip install -r requirements.txt
    Add GROQ_API_KEY=your_key to your .env file

Run:
    uvicorn api:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chatbot import RiseChatbot
from app.llm import is_llm_available, get_llm_info
from app.config import GROQ_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("api")

HERO_PARCELS = {
    "A": {"id": "A", "label": "Parcel A — Heritage", "story": "Heritage Story", "address": "Commerce Street, Montgomery, Alabama 36104", "parcel_id": "11 01 12 4 004 001.000", "acres": 8.34, "zone_context": "heritage", "nearest_anchor": "Rosa Parks Museum", "min_dist_miles": 0.154, "coords": {"lat": 32.3789, "lon": -86.3109}, "rise_score": 74, "urgency": "medium", "top_recommendation": "Civil Rights Heritage Plaza", "open_grants": 2},
    "B": {"id": "B", "label": "Parcel B — IX Hub", "story": "Smart Infrastructure Story", "address": "643 Kimball Street, Montgomery, Alabama 36108", "parcel_id": "11 05 15 1 010 022.000", "acres": 0.33, "zone_context": "ix_hub", "nearest_anchor": "Maxwell AFB Gate", "min_dist_miles": 0.891, "coords": {"lat": 32.3684, "lon": -86.3439}, "rise_score": 29, "urgency": "low", "top_recommendation": "AI Workforce Training Hub", "open_grants": 1},
    "C": {"id": "C", "label": "Parcel C — Food Desert", "story": "Economic Urgency", "address": "Coosa Street, Montgomery, Alabama 36104", "parcel_id": "11 01 12 4 004 001.000", "acres": 8.34, "zone_context": "food_desert", "nearest_anchor": "MGMix Internet Exchange", "min_dist_miles": 0.117, "coords": {"lat": 32.3792, "lon": -86.3087}, "rise_score": 72, "urgency": "high", "top_recommendation": "Community Grocery Co-op", "open_grants": 2},
}

PARCEL_SUMMARY_QUESTIONS = {
    "A": "Give a full summary of Parcel A including its RISE score across all 8 dimensions, its top 3 recommendations, available grants, and community health context.",
    "B": "Give a full summary of Parcel B including its RISE score across all 8 dimensions, its top 3 recommendations, available grants, and workforce context.",
    "C": "Give a full summary of Parcel C including its RISE score across all 8 dimensions, its top 3 recommendations, available grants, and food desert health context.",
}

chatbot: RiseChatbot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    logger.info("Starting RISE RAG API...")
    try:
        chatbot = RiseChatbot()
        if not chatbot.is_ready():
            logger.warning("Knowledge base is empty. Run: python scripts/ingest.py")
        else:
            logger.info("Chatbot ready. Knowledge base: %d chunks.", chatbot.knowledge_base_size())
        if not is_llm_available():
            logger.warning("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")
    except Exception as e:
        logger.error("Failed to initialise chatbot: %s", e)
        chatbot = None
    yield
    logger.info("RISE RAG API shutting down.")


app = FastAPI(
    title="Montgomery RISE RAG API",
    description="RAG-powered chatbot API for Montgomery, Alabama's vacant parcel programme. Powered by ChromaDB, sentence-transformers, and Groq free tier.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Your question about RISE parcels, scoring, grants, or community context.")
    parcel_filter: Optional[str] = Field(default=None, description="Restrict search to one parcel: 'A', 'B', or 'C'.")


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


def _check_chatbot_ready():
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialised.")
    if not chatbot.is_ready():
        raise HTTPException(status_code=503, detail="Knowledge base is empty. Run: python scripts/ingest.py")


def _build_sources(raw_sources: list[dict]) -> list[SourceModel]:
    return [SourceModel(document_title=s.get("document_title", "Unknown"), parcel_id=s.get("parcel_id", "general"), topic=s.get("topic", "general"), source_file=s.get("source_file", "unknown")) for s in raw_sources]


@app.get("/", summary="Health check", tags=["System"])
def health_check():
    kb_size = chatbot.knowledge_base_size() if chatbot else 0
    kb_ready = chatbot.is_ready() if chatbot else False
    llm_info = get_llm_info()
    return {"status": "ok" if kb_ready else "degraded", "service": "Montgomery RISE RAG API", "version": "1.0.0", "knowledge_base_chunks": kb_size, "knowledge_base_ready": kb_ready, "llm_provider": llm_info["provider"], "llm_model": llm_info["model"], "llm_configured": llm_info["configured"], "llm_hosting": llm_info["hosting"], "parcels": len(HERO_PARCELS), "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/parcels", summary="List hero parcels", tags=["Parcels"])
def list_parcels():
    return {"parcels": list(HERO_PARCELS.values()), "total": len(HERO_PARCELS), "generated_at": datetime.utcnow().isoformat() + "Z"}


@app.get("/parcels/{parcel_id}/answer", response_model=ParcelSummaryResponse, summary="Get full AI summary for a hero parcel", tags=["Parcels"])
def answer_parcel(parcel_id: str = Path(..., description="Hero parcel identifier: A, B, or C", example="A")):
    pid = parcel_id.upper().strip()
    if pid not in HERO_PARCELS:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found. Valid IDs: A, B, C")
    _check_chatbot_ready()
    try:
        response = chatbot.ask(PARCEL_SUMMARY_QUESTIONS[pid], parcel_filter=pid)
    except Exception as e:
        logger.error("Error generating parcel summary for %s: %s", pid, e)
        raise HTTPException(status_code=500, detail="Failed to generate parcel summary.")
    p = HERO_PARCELS[pid]
    return ParcelSummaryResponse(parcel_id=pid, label=p["label"], story=p["story"], address=p["address"], acres=p["acres"], zone_context=p["zone_context"], nearest_anchor=p["nearest_anchor"], min_dist_miles=p["min_dist_miles"], rise_score=p["rise_score"], urgency=p["urgency"], top_recommendation=p["top_recommendation"], open_grants=p["open_grants"], summary=response.answer, sources=_build_sources(response.sources), num_chunks_retrieved=response.num_chunks_retrieved, used_fallback=response.used_fallback, generated_at=datetime.utcnow().isoformat() + "Z")


@app.post("/parcels/ask", response_model=AskResponse, summary="Ask a free-form question", tags=["Parcels"])
def ask_question(body: AskRequest):
    _check_chatbot_ready()
    parcel_filter = None
    if body.parcel_filter:
        pf = body.parcel_filter.upper().strip()
        if pf not in {"A", "B", "C"}:
            raise HTTPException(status_code=422, detail="parcel_filter must be 'A', 'B', or 'C'.")
        parcel_filter = pf
    try:
        response = chatbot.ask(body.question, parcel_filter=parcel_filter)
    except Exception as e:
        logger.error("Error generating response: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate a response.")
    return AskResponse(question=response.question, answer=response.answer, sources=_build_sources(response.sources), num_chunks_retrieved=response.num_chunks_retrieved, used_fallback=response.used_fallback, parcel_filter=parcel_filter, generated_at=datetime.utcnow().isoformat() + "Z")
