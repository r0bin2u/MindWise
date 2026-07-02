"""Vector retrieval with neighbor-chunk splicing.

When chunk_i is the top hit, we also pull chunk_{i-n} .. chunk_{i+n} from
the same source and splice them back in original order. This recovers the
surrounding context that a 512-token chunk boundary may have broken,
without having to retrieve at a coarser granularity.

The underlying store (Chroma or FAISS) is chosen by settings.vector_backend;
this module only talks to the store interface.
"""

from __future__ import annotations

from typing import Any

from app.services.vector_store import get_vector_store


def retrieve(query: str, k: int = 3, neighbors: int = 1) -> list[dict[str, Any]]:
    """Return up to k spliced passages for the query.

    Each passage is {"text": spliced_context, "source": source, "hit_idx": idx}.
    Duplicate (source, range) tuples across top-k are collapsed so the LLM
    doesn't see the same paragraph twice.
    """
    if not query or not query.strip():
        return []

    store = get_vector_store()
    try:
        metas = store.query(query, k)
    except Exception:
        return []

    passages: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

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

        spliced = store.fetch_range(source, lo, hi)
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
