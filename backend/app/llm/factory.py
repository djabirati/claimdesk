from app.config import Settings
from app.llm.base import LLMClient, LLMNotConfiguredError


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMNotConfiguredError("GEMINI_API_KEY is not set")
        from app.llm.gemini import GeminiClient

        return GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.llm_model,
            thinking_level=settings.llm_thinking_level,
            timeout_seconds=settings.llm_timeout_seconds,
            embedding_model=settings.embedding_model,
        )
    raise LLMNotConfiguredError(f"Unknown LLM provider: {settings.llm_provider}")
