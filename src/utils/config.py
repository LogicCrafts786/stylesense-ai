"""
Application configuration loader for StyleSense AI.

Uses pydantic-settings to load and validate environment variables from
`.env`, providing a single typed `Settings` object accessible via the
cached `get_settings()` factory. This avoids scattering `os.getenv` calls
throughout the codebase and gives fail-fast validation at startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Strongly typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Gemini API ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model_name: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL_NAME")
    gemini_vision_model_name: str = Field(
        default="gemini-1.5-pro-vision", alias="GEMINI_VISION_MODEL_NAME"
    )

    # --- Weather API ---
    weather_api_key: str = Field(default="", alias="WEATHER_API_KEY")
    weather_api_base_url: str = Field(
        default="https://api.openweathermap.org/data/2.5",
        alias="WEATHER_API_BASE_URL",
    )

    # --- Vector Store ---
    vector_store_type: Literal["chroma", "faiss"] = Field(
        default="chroma", alias="VECTOR_STORE_TYPE"
    )
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    faiss_index_path: str = Field(default="./faiss_index", alias="FAISS_INDEX_PATH")

    # --- Application Settings ---
    app_env: Literal["development", "production", "test"] = Field(
        default="development", alias="APP_ENV"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_conversation_history: int = Field(default=20, alias="MAX_CONVERSATION_HISTORY")
    max_upload_size_mb: int = Field(default=10, alias="MAX_UPLOAD_SIZE_MB")

    # --- Scraper Settings ---
    scraper_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; StyleSenseBot/1.0)",
        alias="SCRAPER_USER_AGENT",
    )
    scraper_timeout_seconds: int = Field(default=10, alias="SCRAPER_TIMEOUT_SECONDS")
    scraper_rate_limit_delay: float = Field(default=1.5, alias="SCRAPER_RATE_LIMIT_DELAY")

    # --- Defaults ---
    default_currency: str = Field(default="USD", alias="DEFAULT_CURRENCY")
    default_locale: str = Field(default="en_US", alias="DEFAULT_LOCALE")

    @field_validator("max_conversation_history", "max_upload_size_mb")
    @classmethod
    def _must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be a positive integer.")
        return value

    def validate_required_for_runtime(self) -> None:
        """
        Explicitly validate that critical secrets are present before the app
        attempts to make live API calls. Called lazily (not at import time)
        so that unit tests can import config without needing a real API key.
        """
        if not self.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set.",
                details="Copy .env.example to .env and set your Gemini API key.",
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Using lru_cache ensures the .env file is parsed once per process,
    improving performance and guaranteeing consistent settings across
    all modules that call this function.
    """
    return Settings()
