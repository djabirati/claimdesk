import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.llm.base import LLMError, LLMResponse

# HTTP codes worth retrying: timeout, quota or rate limit, and server-side errors.
# Other 4xx codes (invalid key, unknown model, malformed request) fail the same way every time.
RETRYABLE_STATUS = {408, 429}


def is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS or status_code >= 500


def build_config(
    *, system: str, schema: type[BaseModel], thinking_level: str | None
) -> types.GenerateContentConfig:
    """Request settings for one structured call.

    No temperature: Google asks not to set sampling parameters on Gemini 3 models (it can
    degrade answers). Answers can therefore vary from one run to the next, which is what
    the evaluation suite measures (step 3).
    """
    thinking = (
        types.ThinkingConfig(thinking_level=types.ThinkingLevel(thinking_level.upper()))
        if thinking_level
        else None
    )
    return types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=thinking,
    )


class GeminiClient:
    def __init__(self, api_key: str, model: str, thinking_level: str | None = None) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.thinking_level = thinking_level

    def generate_json(self, *, system: str, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        config = build_config(system=system, schema=schema, thinking_level=self.thinking_level)
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except errors.APIError as exc:  # quota, invalid key, unknown model, server error...
            raise LLMError(f"Gemini API error: {exc}", retryable=is_retryable(exc.code)) from exc
        except httpx.HTTPError as exc:  # network failure, timeout, proxy refusal
            raise LLMError(f"Network error calling Gemini: {exc}", retryable=True) from exc

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        # "Thinking" tokens are billed as output tokens, so they count in the cost.
        output_tokens = (
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
        )
        return LLMResponse(
            text=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )
