from fastapi import HTTPException, status

from app.config import get_settings
from app.llm.base import LLMClient, LLMNotConfiguredError
from app.llm.factory import build_llm_client


def get_llm() -> LLMClient:
    """FastAPI dependency. Tests replace it with a FakeLLMClient."""
    try:
        return build_llm_client(get_settings())
    except LLMNotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
