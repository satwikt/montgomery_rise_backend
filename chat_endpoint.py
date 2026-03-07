
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

_bot = None

def get_bot():
    global _bot
    if _bot is None:
        from rise_rag.app.chatbot import RiseChatbot  # ← inside function, not at top
        _bot = RiseChatbot()
    return _bot

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    parcel_address: Optional[str] = ""
    parcel_id: Optional[str] = ""

class ChatResponse(BaseModel):
    reply: str
    sources: list[dict] = []
    num_chunks_retrieved: int = 0

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    bot = get_bot()

    parcel_filter: str | None = req.parcel_id.upper() if req.parcel_id in {"A", "B", "C"} else None

    question = req.message
    if req.parcel_address and parcel_filter:
        question = f"[Context: discussing {req.parcel_address} (Parcel {parcel_filter})] {req.message}"
    elif req.parcel_address:
        question = f"[Context: discussing {req.parcel_address}] {req.message}"

    bot.clear_history()
    for msg in req.history:
        bot._history.append({"role": msg.role, "content": msg.content})

    response = bot.ask(question, parcel_filter=parcel_filter)

    return ChatResponse(
        reply=response.answer,
        sources=response.sources,
        num_chunks_retrieved=response.num_chunks_retrieved,
    )