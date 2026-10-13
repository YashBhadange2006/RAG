
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "RAG API"
    app_version: str = "1.0.0"

    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_directory: str = "./data/chroma"
    chunk_size: int = 400
    chunk_overlap: int = 60
    retrieval_top_k: int = 5
    max_upload_mb: int = 10

    # Cosine distance: lower is more similar.
    max_cosine_distance: float = 0.75

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_directory)

        if not path.is_absolute():
            path = ROOT_DIR / path

        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
