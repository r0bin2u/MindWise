"""Build the Chroma knowledge base from markdown / text docs.

  - RecursiveCharacterTextSplitter with chunk_size=512 token, chunk_overlap=64
  - Separators prefer paragraph / sentence ends so chunks don't break mid-sentence
  - Embed with BAAI/bge-small-zh-v1.5 (local, free, Chinese-optimized;
    bge is the Chinese alternative to text-embedding-3-small)
  - Store metadata (source, chunk_idx, total_chunks) so retrieval can splice
    neighboring chunks for extra context — chunk3 hit returns chunk2+3+4

Only LangChain component used here is the splitter; everything else is
straight chromadb / sentence-transformers.
"""
import argparse
import hashlib
import re
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.embeddings import make_embedding_function


# Doc 8.6: noise → embedding drift → recall drops. Minimal cleanup only;
# anything aggressive risks destroying legitimate punctuation / formatting.
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+")


def clean(text: str) -> str:
    t = RE_HTML.sub(" ", text)
    t = RE_URL.sub(" ", t)
    t = t.replace("\u3000", " ").replace("\xa0", " ")
    # collapse only runs of whitespace, keep \n\n paragraph breaks for the splitter
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def make_id(source: str, idx: int) -> str:
    """Deterministic chunk id so re-builds are idempotent."""
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:12]
    return f"{h}_{idx}"


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kb_docs")
    ap.add_argument("--output", default="data/kb")
    ap.add_argument("--collection", default="mindwise_psych")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    ap.add_argument("--embed-backend", default=None,
                    choices=["sentence_transformer", "openai"],
                    help="override EMBEDDING_BACKEND for this build")
    ap.add_argument("--embed-model", default=None,
                    help="override model name within the chosen backend")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop and recreate the collection")
    args = ap.parse_args()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        # 中文优先在段落 / 句子边界切
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    ef = make_embedding_function(
        backend=args.embed_backend, model=args.embed_model
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=args.output)

    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            print(f"dropped existing collection {args.collection}")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=args.collection, embedding_function=ef
    )

    input_root = Path(args.input)
    docs_seen = 0
    chunks_added = 0

    for p in sorted(input_root.rglob("*")):
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        source = str(p.relative_to(input_root))
        raw = p.read_text(encoding="utf-8")
        cleaned = clean(raw)
        chunks = splitter.split_text(cleaned)
        if not chunks:
            continue

        ids = [make_id(source, i) for i in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_idx": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        # upsert so reruns update in place
        col.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        docs_seen += 1
        chunks_added += len(chunks)
        print(f"  [{source}] {len(chunks)} chunks")

    print(f"\ndone. docs={docs_seen} chunks={chunks_added} -> {args.output}/{args.collection}")


if __name__ == "__main__":
    build()
