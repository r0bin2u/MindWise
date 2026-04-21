"""Chroma retrieval with neighbor-chunk splicing.

Per the design image: when chunk_i is the top hit, we also pull chunk_{i-n}
.. chunk_{i+n} from the same source and splice them back in original order.
This recovers the surrounding context that a 512-token chunk boundary may
have broken, without having to retrieve at a coarser granularity.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.core.config import settings


DEFAULT_COLLECTION = "mindwise_psych"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=DEFAULT_EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=DEFAULT_COLLECTION, embedding_function=ef
    )


def _fetch_neighbors(col, source: str, lo: int, hi: int) -> list[dict]:
    """Return all chunks from `source` with chunk_idx in [lo, hi], sorted."""
    where = {
        "$and": [
            {"source": source},
            {"chunk_idx": {"$gte": lo}},
            {"chunk_idx": {"$lte": hi}},
        ]
    }
    got = col.get(where=where, include=["documents", "metadatas"])
    pairs = list(zip(got["documents"], got["metadatas"]))
    pairs.sort(key=lambda p: p[1]["chunk_idx"])
    return [{"doc": d, "meta": m} for d, m in pairs]


def retrieve(query: str, k: int = 3, neighbors: int = 1) -> list[dict[str, Any]]:
    """Return up to k spliced passages for the query.

    Each passage is {"text": spliced_context, "source": source, "hit_idx": idx}.
    Duplicate (source, range) tuples across top-k are collapsed so the LLM
    doesn't see the same paragraph twice.
    """
    if not query or not query.strip():
        return []

    col = _get_collection()
    try:
        res = col.query(
            query_texts=[query],
            n_results=k,
            include=["metadatas", "distances"],
        )
    except Exception:
        return []

    passages: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    metas = res.get("metadatas", [[]])[0]
    for meta in metas:
        source = meta["source"]
        idx = int(meta["chunk_idx"])
        total = int(meta["total_chunks"])
        lo = max(0, idx - neighbors)
        hi = min(total - 1, idx + neighbors)

        key = (source, lo, hi)
        if key in seen:
            continue
        seen.add(key)

        spliced = _fetch_neighbors(col, source, lo, hi)
        if not spliced:
            continue
        text = " ".join(s["doc"] for s in spliced)
        passages.append({"text": text, "source": source, "hit_idx": idx})

    return passages


def format_passages(passages: list[dict[str, Any]]) -> str:
    """Render passages for inclusion in an LLM prompt."""
    if not passages:
        return "(未检索到相关资料)"
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[资料 {i} · 来源: {p['source']}]\n{p['text']}")
    return "\n\n".join(parts)
