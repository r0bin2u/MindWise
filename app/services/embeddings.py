"""Embedding function factory — single source of truth for both KB build
and retrieval so the vectors stay compatible.

Two supported backends:

**sentence_transformer (default)** — BAAI/bge-small-zh-v1.5
  - Runs locally on CPU/GPU, no API calls
  - ~100MB weights, Chinese-optimized, 512-dim vectors
  - Matches the project narrative ("零 API 费用, 本地离线部署")

**openai** — text-embedding-3-small
  - Zero deployment overhead, no model download
  - Higher semantic precision on general Chinese text per OpenAI benchmarks
  - 1536-dim vectors, costs $0.02 / 1M tokens, requires outbound network
  - Pick this when: accuracy > cost, or you already have an OpenAI key
    and don't want to manage local weights

**Critical consistency rule**: the backend used to BUILD the Chroma index
must match the one used at query time. Vectors from different models live
in different spaces — mixing them silently breaks retrieval. If you switch
backends, drop `data/kb/` and rebuild (`python -m scripts.build_kb --rebuild`).
"""

from functools import lru_cache

from chromadb.utils import embedding_functions

from app.core.config import settings


BACKEND_SENTENCE_TRANSFORMER = "sentence_transformer"
BACKEND_OPENAI = "openai"

DEFAULT_ST_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def _resolve_backend(backend: str | None) -> str:
    b = (backend or settings.embedding_backend or BACKEND_SENTENCE_TRANSFORMER).lower()
    # allow users to write the model name directly as the backend
    if b in ("openai", "text-embedding-3-small", "text-embedding-3-large"):
        return BACKEND_OPENAI
    if b in ("sentence_transformer", "st", "bge", "local"):
        return BACKEND_SENTENCE_TRANSFORMER
    raise ValueError(
        f"unknown embedding backend: {backend!r}. Expected 'sentence_transformer' or 'openai'."
    )


def make_embedding_function(
    backend: str | None = None,
    model: str | None = None,
):
    """Return a Chroma-compatible embedding function.

    Priority: explicit args > env settings > hard-coded defaults.
    """
    kind = _resolve_backend(backend)

    if kind == BACKEND_OPENAI:
        if not settings.openai_api_key:
            raise RuntimeError(
                "embedding_backend=openai but OPENAI_API_KEY is not set; "
                "add it to .env or switch EMBEDDING_BACKEND back to "
                "sentence_transformer."
            )
        kwargs = {
            "api_key": settings.openai_api_key,
            "model_name": model or settings.embedding_model or DEFAULT_OPENAI_MODEL,
        }
        if settings.openai_base_url:
            kwargs["api_base"] = settings.openai_base_url
        return embedding_functions.OpenAIEmbeddingFunction(**kwargs)

    # sentence_transformer path
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model or settings.embedding_model or DEFAULT_ST_MODEL
    )


# ---------------------------------------------------------------------------
# Raw vectors — the FAISS / Milvus backends need the embeddings in hand rather
# than delegating to Chroma's internal embedding call. Vectors are always
# L2-normalised so an inner-product index behaves as cosine similarity.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _l2_normalize(vectors) -> list[list[float]]:
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


def embed_texts(texts, backend: str | None = None, model: str | None = None) -> list[list[float]]:
    """Return L2-normalised embedding vectors for `texts`.

    Uses the same backend resolution as make_embedding_function, so the
    vectors match whatever the KB was built with.
    """
    kind = _resolve_backend(backend)
    items = list(texts)

    if kind == BACKEND_SENTENCE_TRANSFORMER:
        m = _st_model(model or settings.embedding_model or DEFAULT_ST_MODEL)
        return m.encode(items, normalize_embeddings=True).tolist()

    # openai — reuse the Chroma embedding function purely as an embedder,
    # then normalise ourselves (the API vectors aren't guaranteed unit-norm).
    ef = make_embedding_function(backend, model)
    return _l2_normalize(ef(items))
