"""
Add this to your existing api.py (or mount as a router).

Connects the frontend ChatArea → POST /chat → RiseChatbot RAG pipeline.

Required: RiseChatbot must be initialised (knowledge base ingested).
Run once before starting the server:
    python scripts/ingest.py
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.chatbot import RiseChatbot

router = APIRouter()

# ─── Initialise once at startup (shared across all requests) ──────────────────
# This loads ChromaDB embeddings into memory — do it once, not per request.
_bot: RiseChatbot | None = None

def get_bot() -> RiseChatbot:
    global _bot
    if _bot is None:
        _bot = RiseChatbot()
    return _bot


# ─── Request / Response schemas ───────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    parcel_address: Optional[str] = ""
    parcel_id: Optional[str] = ""   # "A", "B", "C" or ""

class ChatResponse(BaseModel):
    reply: str
    sources: list[dict] = []
    num_chunks_retrieved: int = 0


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    bot = get_bot()

    # Map parcel_id → filter understood by RiseChatbot ("A" | "B" | "C" | None)
    parcel_filter: str | None = req.parcel_id.upper() if req.parcel_id in {"A", "B", "C"} else None

    # Inject parcel context into the question if a parcel is selected
    question = req.message
    if req.parcel_address and parcel_filter:
        question = f"[Context: discussing {req.parcel_address} (Parcel {parcel_filter})] {req.message}"
    elif req.parcel_address:
        question = f"[Context: discussing {req.parcel_address}] {req.message}"

    # Pass conversation history into the bot so it has multi-turn context.
    # RiseChatbot.ask() uses its internal history list — we sync it from the
    # frontend's history so the API stays stateless.
    bot.clear_history()
    for msg in req.history:
        # Replay history so the bot has full conversation context
        bot._history.append({
            "role": msg.role,
            "content": msg.content,
        })

    response = bot.ask(question, parcel_filter=parcel_filter)

    return ChatResponse(
        reply=response.answer,
        sources=response.sources,
        num_chunks_retrieved=response.num_chunks_retrieved,
    )
