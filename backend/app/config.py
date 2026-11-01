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

    # Queue and worker.
    # How many times a job may run before it is marked failed (provider errors are retried).
    job_max_attempts: int = 4
    # Delay before the first retry, multiplied by 4 at each new failure (15 s, 1 min, 4 min).
    job_retry_base_seconds: float = 15
    # A job still "running" after this long is considered abandoned (worker crashed or killed).
    job_lock_timeout_seconds: float = 300
    # How long an idle worker waits before looking at the queue again.
    worker_poll_seconds: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
