import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.llm.base import LLMError, LLMResponse


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def generate_json(self, *, system: str, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except errors.APIError as exc:  # quota, invalid key, server error...
            raise LLMError(f"Gemini API error: {exc}") from exc
        except httpx.HTTPError as exc:  # network failure, timeout, proxy refusal
            raise LLMError(f"Network error calling Gemini: {exc}") from exc

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
