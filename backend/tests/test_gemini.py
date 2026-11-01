"""The Gemini client's settings and error mapping, checked without calling the API."""

import pytest
from google.genai import types

from app.config import Settings
from app.llm.gemini import build_config, is_retryable
from app.schemas import TriageOutput


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [
        (429, True),  # quota or rate limit
        (408, True),  # timeout
        (500, True),
        (503, True),  # overloaded
        (400, False),  # malformed request
        (401, False),  # invalid key
        (403, False),  # key without access
        (404, False),  # unknown model, or model closed to new users
    ],
)
def test_only_temporary_errors_are_retryable(status_code: int, retryable: bool) -> None:
    assert is_retryable(status_code) is retryable


def test_config_sends_no_temperature_and_asks_for_json() -> None:
    config = build_config(system="sys", schema=TriageOutput, thinking_level="low")
    assert config.temperature is None  # Google asks not to set it on Gemini 3 models
    assert config.response_mime_type == "application/json"
    assert config.response_schema is TriageOutput
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_no_thinking_setting_is_sent_when_none_is_configured() -> None:
    config = build_config(system="sys", schema=TriageOutput, thinking_level=None)
    assert config.thinking_config is None


def test_empty_thinking_level_in_env_means_none() -> None:
    settings = Settings(_env_file=None, llm_thinking_level="")  # type: ignore[call-arg, arg-type]
    assert settings.llm_thinking_level is None
