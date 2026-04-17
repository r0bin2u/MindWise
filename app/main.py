from fastapi import FastAPI

from app.api import audio, video
from app.core.config import settings

app = FastAPI(title="MindWise", version="0.1.0")

app.include_router(audio.router)
app.include_router(video.router)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}
