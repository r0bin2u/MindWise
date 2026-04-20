"""Shared async Ollama client. Avoids instantiating a new client per call."""
from ollama import AsyncClient

from app.core.config import settings


_client: AsyncClient | None = None


def get_client() -> AsyncClient:
    global _client
    if _client is None:
        _client = AsyncClient(host=settings.ollama_host)
    return _client
