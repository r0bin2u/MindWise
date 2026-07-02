import pytest

from app.services import vector_store as vs


def test_make_store_unknown_backend():
    with pytest.raises(ValueError):
        vs._make_store("weaviate")


# FAISS is an optional backend; skip cleanly when the extra isn't installed.
pytest.importorskip("faiss")


def _meta(idx, total=3, source="s.md"):
    return {"source": source, "chunk_idx": idx, "total_chunks": total}


def test_faiss_roundtrip(tmp_path, monkeypatch):
    """Build → persist → reload → query/fetch_range, with deterministic
    fake vectors so the test stays offline and fast (no model download)."""
    from app.services import embeddings as emb

    table = {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0, 0.0], "c": [0.0, 0.0, 1.0, 0.0]}
    monkeypatch.setattr(
        emb, "embed_texts", lambda texts, backend=None, model=None: [table[t] for t in texts]
    )

    store = vs.FaissStore(str(tmp_path))
    store.upsert(["s_0", "s_1", "s_2"], ["a", "b", "c"], [_meta(0), _meta(1), _meta(2)])
    store.persist()

    # a fresh instance must read everything back from disk
    reader = vs.FaissStore(str(tmp_path))
    hits = reader.query("b", k=1)
    assert hits and hits[0]["chunk_idx"] == 1

    rng = reader.fetch_range("s.md", 0, 2)
    assert [r["meta"]["chunk_idx"] for r in rng] == [0, 1, 2]
    assert rng[1]["doc"] == "b"


def test_faiss_fetch_range_clamps_missing(tmp_path, monkeypatch):
    from app.services import embeddings as emb

    monkeypatch.setattr(
        emb, "embed_texts", lambda texts, backend=None, model=None: [[1.0, 0.0]] * len(texts)
    )

    store = vs.FaissStore(str(tmp_path))
    store.upsert(["s_0"], ["only"], [_meta(0, total=1)])
    store.persist()

    reader = vs.FaissStore(str(tmp_path))
    # asking beyond the available range returns just what's present
    rng = reader.fetch_range("s.md", 0, 5)
    assert [r["meta"]["chunk_idx"] for r in rng] == [0]


# Milvus needs both the client lib and a running standalone server; skip
# gracefully when either is missing so the suite stays runnable everywhere.
def test_milvus_roundtrip(monkeypatch):
    pytest.importorskip("pymilvus")
    from app.core.config import settings
    from app.services import embeddings as emb

    table = {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0, 0.0], "c": [0.0, 0.0, 1.0, 0.0]}
    monkeypatch.setattr(
        emb, "embed_texts", lambda texts, backend=None, model=None: [table[t] for t in texts]
    )

    store = vs.MilvusStore(uri=settings.milvus_uri, collection="mindwise_test_rt")
    try:
        store.drop()
        store.upsert(["s_0", "s_1", "s_2"], ["a", "b", "c"], [_meta(0), _meta(1), _meta(2)])
        store.persist()
    except Exception as exc:  # no server reachable
        pytest.skip(f"Milvus not reachable: {exc}")

    try:
        hits = store.query("b", k=1)
        assert hits and hits[0]["chunk_idx"] == 1
        rng = store.fetch_range("s.md", 0, 2)
        assert [r["meta"]["chunk_idx"] for r in rng] == [0, 1, 2]
        assert rng[1]["doc"] == "b"
    finally:
        store.drop()
