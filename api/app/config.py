from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App settings, overridable with TERRASENSE_* env vars or an .env file."""

    model_config = SettingsConfigDict(env_prefix="TERRASENSE_", env_file=".env", extra="ignore")

    # sqlite cho dev, docker-compose sẽ set sang postgres
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'terrasense.db').as_posix()}"
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.onnx"
    meta_path: Path = PROJECT_ROOT / "artifacts" / "model_meta.json"
    max_upload_mb: float = 5.0
    max_batch_size: int = 16
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
