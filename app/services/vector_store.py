"""Pluggable vector-store backend behind a single small interface.

Two backends, selected by `settings.vector_backend`:

**chroma (default)** — ChromaDB PersistentClient. Embeds internally via the
  Chroma embedding function; ANN index is hnswlib's HNSW. This is the
  original path, unchanged in behaviour.

**faiss** — a raw `faiss.IndexHNSWFlat` (HNSW graph living inside FAISS)
  with inner-product metric. Vectors are L2-normalised at embed time so
  inner product == cosine similarity. FAISS only holds vectors + the graph,
  so chunk text and metadata (source / chunk_idx / total_chunks) live in a
  JSON sidecar keyed by row order; neighbour splicing reads the sidecar.

Every store exposes the same four methods so `retrieval.py` and
`build_kb.py` don't care which one is live:

    upsert(ids, documents, metadatas) -> None
    query(text, k)                    -> list[meta dict]
    fetch_range(source, lo, hi)       -> list[{"doc", "meta"}] sorted by idx
    persist()                         -> None   (chroma writes through; faiss dumps files)

Build and query MUST use the same backend — a FAISS index and a Chroma
collection are not interchangeable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings


DEFAULT_COLLECTION = "mindwise_psych"

# HNSW graph params for the FAISS backend. M = neighbours per node;
# efConstruction / efSearch trade build+query time for recall.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


# ---------------------------------------------------------------------------
# chroma
# ---------------------------------------------------------------------------


class ChromaStore:
    def __init__(self, path: str, collection: str, embed_backend=None, embed_model=None):
        import chromadb

        from app.services.embeddings import make_embedding_function

        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._ef = make_embedding_function(backend=embed_backend, model=embed_model)
        self._col = self._client.get_or_create_collection(
            name=collection, embedding_function=self._ef
        )

    def upsert(self, ids, documents, metadatas) -> None:
        self._col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, text: str, k: int) -> list[dict[str, Any]]:
        res = self._col.query(
            query_texts=[text], n_results=k, include=["metadatas", "distances"]
        )
        return res.get("metadatas", [[]])[0]

    def fetch_range(self, source: str, lo: int, hi: int) -> list[dict[str, Any]]:
        where = {
            "$and": [
                {"source": source},
                {"chunk_idx": {"$gte": lo}},
                {"chunk_idx": {"$lte": hi}},
            ]
        }
        got = self._col.get(where=where, include=["documents", "metadatas"])
        pairs = list(zip(got["documents"], got["metadatas"]))
        pairs.sort(key=lambda p: p[1]["chunk_idx"])
        return [{"doc": d, "meta": m} for d, m in pairs]

    def drop(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, embedding_function=self._ef
        )

    def persist(self) -> None:
        # PersistentClient writes through on every upsert.
        pass


# ---------------------------------------------------------------------------
# faiss (IndexHNSWFlat + JSON sidecar)
# ---------------------------------------------------------------------------


class FaissStore:
    def __init__(self, path: str, embed_backend=None, embed_model=None):
        self._dir = Path(path)
        self._embed_backend = embed_backend
        self._embed_model = embed_model
        self._buf: dict[str, tuple] = {}  # id -> (vector, document, meta) during build
        self._index = None
        self._rows: list[dict] = []  # row i corresponds to vector i in the index
        self._by_src_idx: dict[tuple[str, int], dict] = {}
        self._loaded = False

    # ---- build path ----
    def upsert(self, ids, documents, metadatas) -> None:
        from app.services.embeddings import embed_texts

        vecs = embed_texts(documents, backend=self._embed_backend, model=self._embed_model)
        for i, _id in enumerate(ids):
            self._buf[_id] = (vecs[i], documents[i], metadatas[i])

    def persist(self) -> None:
        import faiss
        import numpy as np

        if not self._buf:
            return
        items = list(self._buf.items())
        dim = len(items[0][1][0])
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        mat = np.asarray([v for _, (v, _d, _m) in items], dtype="float32")
        index.add(mat)

        rows = [{"id": _id, "document": d, **m} for _id, (_v, d, m) in items]
        self._dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._dir / "index.faiss"))
        (self._dir / "meta.json").write_text(
            json.dumps({"dim": dim, "rows": rows}, ensure_ascii=False), encoding="utf-8"
        )

    # ---- query path ----
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import faiss

        idx_path = self._dir / "index.faiss"
        meta_path = self._dir / "meta.json"
        if not idx_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"FAISS index not found under {self._dir}; "
                "build it first with `python -m scripts.build_kb --vector-backend faiss`."
            )
        self._index = faiss.read_index(str(idx_path))
        self._index.hnsw.efSearch = HNSW_EF_SEARCH
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._rows = meta["rows"]
        self._by_src_idx = {(r["source"], int(r["chunk_idx"])): r for r in self._rows}
        self._loaded = True

    def query(self, text: str, k: int) -> list[dict[str, Any]]:
        import numpy as np

        from app.services.embeddings import embed_texts

        self._ensure_loaded()
        qv = np.asarray(
            embed_texts([text], backend=self._embed_backend, model=self._embed_model),
            dtype="float32",
        )
        _dist, idxs = self._index.search(qv, k)
        out = []
        for ridx in idxs[0]:
            if ridx == -1:
                continue
            r = self._rows[ridx]
            out.append(
                {
                    "source": r["source"],
                    "chunk_idx": int(r["chunk_idx"]),
                    "total_chunks": int(r["total_chunks"]),
                }
            )
        return out

    def fetch_range(self, source: str, lo: int, hi: int) -> list[dict[str, Any]]:
        self._ensure_loaded()
        out = []
        for i in range(lo, hi + 1):
            r = self._by_src_idx.get((source, i))
            if r is None:
                continue
            out.append(
                {
                    "doc": r["document"],
                    "meta": {
                        "source": r["source"],
                        "chunk_idx": int(r["chunk_idx"]),
                        "total_chunks": int(r["total_chunks"]),
                    },
                }
            )
        return out

    def drop(self) -> None:
        self._buf = {}
        for f in ("index.faiss", "meta.json"):
            p = self._dir / f
            if p.exists():
                p.unlink()
        self._loaded = False


# ---------------------------------------------------------------------------
# milvus (standalone, IVF_FLAT index = FAISS engine)
# ---------------------------------------------------------------------------


class MilvusStore:
    """Distributed vector DB backend. Vectors + scalar metadata live in one
    collection; neighbour splicing uses a boolean-expression scalar filter
    instead of a separate sidecar. IVF_FLAT is FAISS-backed inside Milvus's
    Knowhere engine.

    Like FaissStore, build buffers in memory and materialises the whole
    collection in persist() (drop-and-recreate), so a rebuild is a full,
    deterministic reload rather than in-place mutation.
    """

    def __init__(
        self,
        uri: str,
        collection: str,
        *,
        index_type: str | None = None,
        metric: str | None = None,
        nlist: int | None = None,
        nprobe: int | None = None,
        embed_backend=None,
        embed_model=None,
    ):
        self._uri = uri
        self._name = collection
        self._index_type = index_type or settings.milvus_index_type
        self._metric = metric or settings.milvus_metric
        self._nlist = nlist or settings.milvus_nlist
        self._nprobe = nprobe or settings.milvus_nprobe
        self._embed_backend = embed_backend
        self._embed_model = embed_model
        self._buf: dict[str, tuple] = {}  # id -> (vector, document, meta) during build
        self._client = None
        self._loaded = False

    def _c(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self._uri)
        return self._client

    # ---- build path ----
    def upsert(self, ids, documents, metadatas) -> None:
        from app.services.embeddings import embed_texts

        vecs = embed_texts(documents, backend=self._embed_backend, model=self._embed_model)
        for i, _id in enumerate(ids):
            self._buf[_id] = (vecs[i], documents[i], metadatas[i])

    def persist(self) -> None:
        from pymilvus import DataType

        if not self._buf:
            return
        items = list(self._buf.items())
        dim = len(items[0][1][0])
        client = self._c()

        if client.has_collection(self._name):
            client.drop_collection(self._name)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("total_chunks", DataType.INT64)
        schema.add_field("document", DataType.VARCHAR, max_length=8192)
        client.create_collection(collection_name=self._name, schema=schema)

        rows = [
            {
                "id": _id,
                "vector": v,
                "source": m["source"],
                "chunk_idx": int(m["chunk_idx"]),
                "total_chunks": int(m["total_chunks"]),
                "document": d,
            }
            for _id, (v, d, m) in items
        ]
        client.insert(collection_name=self._name, data=rows)
        client.flush(collection_name=self._name)

        # IVF k-means needs at least nlist training points; clamp for tiny KBs.
        nlist = max(1, min(self._nlist, len(rows)))
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self._index_type,
            metric_type=self._metric,
            params={"nlist": nlist},
        )
        client.create_index(collection_name=self._name, index_params=index_params)
        client.load_collection(collection_name=self._name)
        self._loaded = True

    # ---- query path ----
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        client = self._c()
        if not client.has_collection(self._name):
            raise RuntimeError(
                f"Milvus collection {self._name!r} not found; "
                "build it first with `python -m scripts.build_kb --vector-backend milvus`."
            )
        client.load_collection(collection_name=self._name)
        self._loaded = True

    def query(self, text: str, k: int) -> list[dict[str, Any]]:
        from app.services.embeddings import embed_texts

        self._ensure_loaded()
        qv = embed_texts([text], backend=self._embed_backend, model=self._embed_model)
        nprobe = max(1, min(self._nprobe, self._nlist))
        res = self._c().search(
            collection_name=self._name,
            data=qv,
            limit=k,
            search_params={"metric_type": self._metric, "params": {"nprobe": nprobe}},
            output_fields=["source", "chunk_idx", "total_chunks"],
        )
        out = []
        for hit in res[0]:
            e = hit["entity"]
            out.append(
                {
                    "source": e["source"],
                    "chunk_idx": int(e["chunk_idx"]),
                    "total_chunks": int(e["total_chunks"]),
                }
            )
        return out

    def fetch_range(self, source: str, lo: int, hi: int) -> list[dict[str, Any]]:
        self._ensure_loaded()
        expr = f'source == "{source}" and chunk_idx >= {lo} and chunk_idx <= {hi}'
        rows = self._c().query(
            collection_name=self._name,
            filter=expr,
            output_fields=["document", "source", "chunk_idx", "total_chunks"],
        )
        rows.sort(key=lambda r: r["chunk_idx"])
        return [
            {
                "doc": r["document"],
                "meta": {
                    "source": r["source"],
                    "chunk_idx": int(r["chunk_idx"]),
                    "total_chunks": int(r["total_chunks"]),
                },
            }
            for r in rows
        ]

    def drop(self) -> None:
        self._buf = {}
        client = self._c()
        if client.has_collection(self._name):
            client.drop_collection(self._name)
        self._loaded = False


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def _make_store(
    backend: str | None,
    *,
    path: str | None = None,
    collection: str | None = None,
    embed_backend: str | None = None,
    embed_model: str | None = None,
):
    b = (backend or settings.vector_backend or "chroma").lower()
    if b == "chroma":
        return ChromaStore(
            path or settings.chroma_path,
            collection or DEFAULT_COLLECTION,
            embed_backend,
            embed_model,
        )
    if b == "faiss":
        return FaissStore(path or settings.faiss_path, embed_backend, embed_model)
    if b == "milvus":
        return MilvusStore(
            uri=path or settings.milvus_uri,
            collection=collection or settings.milvus_collection,
            embed_backend=embed_backend,
            embed_model=embed_model,
        )
    raise ValueError(
        f"unknown vector_backend: {backend!r}. Expected 'chroma', 'faiss' or 'milvus'."
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Cached store for the query path, configured from settings."""
    return _make_store(settings.vector_backend)


def get_build_store(
    backend: str | None = None,
    *,
    path: str | None = None,
    collection: str | None = None,
    embed_backend: str | None = None,
    embed_model: str | None = None,
):
    """Fresh (uncached) store for build_kb, with per-build overrides."""
    return _make_store(
        backend,
        path=path,
        collection=collection,
        embed_backend=embed_backend,
        embed_model=embed_model,
    )
