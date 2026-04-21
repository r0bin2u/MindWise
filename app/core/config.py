from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-7b-psychqa"
    base_model: str = "qwen2.5:7b"

    chroma_path: str = "./data/kb"
    excel_log_path: str = "./data/consult_log.xlsx"

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
