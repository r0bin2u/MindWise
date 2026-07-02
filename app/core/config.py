from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-7b-psychqa"
    base_model: str = "qwen2.5:7b"

    chroma_path: str = "./data/kb"
    excel_log_path: str = "./data/consult_log.xlsx"

    # Vector store backend: "chroma" (default, hnswlib HNSW), "faiss"
    # (faiss.IndexHNSWFlat) or "milvus" (standalone, IVF_FLAT). Build and
    # query must use the same backend — the indexes are not interchangeable.
    vector_backend: str = "chroma"
    faiss_path: str = "./data/kb_faiss"

    # Milvus backend (only used when vector_backend=milvus). IVF_FLAT is
    # FAISS-backed inside Milvus; nlist is clamped to the corpus size at
    # build time so tiny KBs still train.
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "mindwise_psych"
    milvus_index_type: str = "IVF_FLAT"
    milvus_metric: str = "COSINE"
    milvus_nlist: int = 128
    milvus_nprobe: int = 16

    # Embedding backend for Chroma. "sentence_transformer" (default) runs
    # bge-small-zh-v1.5 locally; "openai" calls text-embedding-3-small.
    # Must stay consistent between KB build and retrieval — vectors from
    # different models have different dimensions / semantics.
    embedding_backend: str = "sentence_transformer"
    embedding_model: str = ""  # empty = backend default
    openai_api_key: str = ""
    openai_base_url: str = ""  # optional, for Azure / proxies

    smtp_host: str = ""
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    alert_to: str = ""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
