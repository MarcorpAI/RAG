from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_api_key: str | None = Field(default=None, alias="HF_API_KEY")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    generation_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", alias="GENERATION_MODEL"
    )
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE", gt=0)
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP", ge=0)
    top_k_default: int = Field(default=5, alias="TOP_K_DEFAULT", gt=0)
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB", gt=0)
    hf_timeout_seconds: int = Field(default=60, alias="HF_TIMEOUT_SECONDS", gt=0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_size(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
