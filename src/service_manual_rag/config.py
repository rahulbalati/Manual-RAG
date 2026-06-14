"""Application settings via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_pdf: Path = Field(default=Path("data/raw/MX-B468P-Service-Manual.pdf"))
    processed_dir: Path = Field(default=Path("data/processed"))
    assets_dir: Path = Field(default=Path("assets"))
    index_dir: Path = Field(default=Path("data/index"))

    hybrid_search_enabled: bool = Field(default=True)
    hybrid_semantic_weight: float = Field(default=0.6, ge=0.0)
    hybrid_keyword_weight: float = Field(default=0.4, ge=0.0)
    hybrid_rrf_k: int = Field(default=60, ge=1)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def frontend_dir(self) -> Path:
        return self.project_root / "frontend"


@lru_cache
def get_settings() -> Settings:
    return Settings()
