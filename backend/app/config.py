from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a local .env file."""

    # Works from the repo root or from backend/ (the .env file lives at the repo root).
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "postgresql+psycopg://claimdesk:claimdesk@localhost:5432/claimdesk"

    llm_provider: Literal["gemini"] = "gemini"
    llm_model: str = "gemini-3.8-flash"
    gemini_api_key: str | None = None
    # How much the model "thinks" before answering (Gemini 3: low, medium or high).
    # Sorting an email does not need long reasoning, and thinking tokens are billed.
    # Leave empty for a Gemini 2.5 model, which does not accept this setting.
    llm_thinking_level: Literal["low", "medium", "high"] | None = "low"

    # Used only to estimate the cost of each run (USD per million tokens).
    # Gemini 3.8 Flash paid-tier prices; Google announced they double on 1 January 2027
    # (1.50 and 7.50). Check the provider's current pricing and override in .env if needed.
    llm_input_price_per_mtok: float = 0.75
    llm_output_price_per_mtok: float = 3.75

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

    @field_validator("llm_thinking_level", mode="before")
    @classmethod
    def empty_means_none(cls, value: object) -> object:
        # LLM_THINKING_LEVEL= (empty) in .env means "do not send the setting".
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
