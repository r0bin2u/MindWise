import pytest

from app.services.embeddings import (
    BACKEND_OPENAI,
    BACKEND_SENTENCE_TRANSFORMER,
    _resolve_backend,
    make_embedding_function,
)


# ---------------- _resolve_backend ----------------

def test_resolve_backend_defaults_to_sentence_transformer():
    assert _resolve_backend(None) == BACKEND_SENTENCE_TRANSFORMER


def test_resolve_backend_sentence_transformer_aliases():
    for name in ["sentence_transformer", "ST", "bge", "local"]:
        assert _resolve_backend(name) == BACKEND_SENTENCE_TRANSFORMER


def test_resolve_backend_openai_aliases():
    for name in ["openai", "text-embedding-3-small", "text-embedding-3-large"]:
        assert _resolve_backend(name) == BACKEND_OPENAI


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        _resolve_backend("some-random-model")


# ---------------- make_embedding_function ----------------

def test_make_embedding_function_openai_needs_api_key(monkeypatch):
    """Requesting the openai backend without a key should fail loudly —
    better to see a clear RuntimeError at build time than mysterious
    retrieval errors after the KB is already populated."""
    from app.services import embeddings as mod
    monkeypatch.setattr(mod.settings, "openai_api_key", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_embedding_function(backend="openai")


def test_make_embedding_function_st_default_works():
    # only imports + constructs the function; does NOT load the model file
    # (SentenceTransformerEmbeddingFunction downloads on first encode, not
    # on init), so this is fast and offline-safe.
    ef = make_embedding_function(backend="sentence_transformer")
    assert ef is not None
    assert hasattr(ef, "__call__")
