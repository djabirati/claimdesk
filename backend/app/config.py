from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a local .env file."""

    # Works from the repo root or from backend/ (the .env file lives at the repo root).
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "postgresql+psycopg://claimdesk:claimdesk@localhost:5432/claimdesk"

    llm_provider: Literal["gemini"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None

    # Used only to estimate the cost of each run (USD per million tokens).
    # Check the provider's current pricing and override in .env if needed.
    llm_input_price_per_mtok: float = 0.30
    llm_output_price_per_mtok: float = 2.50

    # Number of LLM calls allowed for one email when the output fails validation.
    triage_max_attempts: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
