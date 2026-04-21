from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import audio, chat, fusion, intent, rag, video
from app.core.config import settings
from app.core.tracing import flush as langfuse_flush
from app.core.tracing import init_langfuse

app = FastAPI(title="MindWise", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    init_langfuse()


@app.on_event("shutdown")
async def _shutdown() -> None:
    # give Langfuse a chance to push pending traces before the process dies
    langfuse_flush()


# Auto-instrument HTTP layer: latency histogram, request count by status,
# in-progress gauge. Exposed at /metrics. Business counters live in
# app.core.metrics and get incremented from the pipeline itself.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(audio.router)
app.include_router(video.router)
app.include_router(fusion.router)
app.include_router(intent.router)
app.include_router(rag.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}
