from functools import lru_cache
from typing import Literal, Self

from pydantic import field_validator, model_validator
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
    # Longest wait for one model call. Without it a call that never returns blocks the
    # worker for good (the SDK sets no limit). Calls of 40 s were seen on the free tier.
    llm_timeout_seconds: float = 90

    # Used only to estimate the cost of each run (USD per million tokens).
    # Gemini 3.8 Flash paid-tier prices; Google announced they double on 1 January 2027
    # (1.50 and 7.50). Check the provider's current pricing and override in .env if needed.
    llm_input_price_per_mtok: float = 0.75
    llm_output_price_per_mtok: float = 3.75

    # Number of LLM calls allowed for one email when the output fails validation.
    triage_max_attempts: int = 2
    # Most model calls in one agent run, so a confused agent cannot loop (and spend) forever.
    agent_max_steps: int = 8

    # Queue and worker.
    # How many times a job may run before it is marked failed (provider errors are retried).
    job_max_attempts: int = 4
    # Delay before the first retry, multiplied by 4 at each new failure (15 s, 1 min, 4 min).
    job_retry_base_seconds: float = 15
    # A job still "running" after this long is considered abandoned (worker crashed or killed).
    # It must outlast the longest job: an agent run of agent_max_steps calls (see below).
    job_lock_timeout_seconds: float = 900
    # How long an idle worker waits before looking at the queue again.
    worker_poll_seconds: float = 1.0

    @field_validator("llm_thinking_level", mode="before")
    @classmethod
    def empty_means_none(cls, value: object) -> object:
        # LLM_THINKING_LEVEL= (empty) in .env means "do not send the setting".
        return None if value == "" else value

    @model_validator(mode="after")
    def jobs_end_before_the_lock_expires(self) -> Self:
        # A triage makes up to triage_max_attempts model calls, an agent run up to
        # agent_max_steps. If a job can last longer than the lock timeout, a job still
        # waiting for the model is handed to a second worker.
        calls = max(self.triage_max_attempts, self.agent_max_steps)
        longest_job = self.llm_timeout_seconds * calls
        if longest_job >= self.job_lock_timeout_seconds:
            raise ValueError(
                f"llm_timeout_seconds x {calls} model calls ({longest_job:g} s) must be "
                f"below job_lock_timeout_seconds ({self.job_lock_timeout_seconds:g} s)"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
