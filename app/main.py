from fastapi import FastAPI

from app.api import audio, chat, fusion, intent, rag, video
from app.core.config import settings

app = FastAPI(title="MindWise", version="0.1.0")

app.include_router(audio.router)
app.include_router(video.router)
app.include_router(fusion.router)
app.include_router(intent.router)
app.include_router(rag.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}
