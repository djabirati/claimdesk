from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.llm.base import (
    EMBEDDING_DIMENSIONS,
    LLMError,
    LLMResponse,
    Message,
    ModelTurn,
    ToolCall,
    ToolResults,
    ToolSpec,
    UserMessage,
    normalize,
)

T = TypeVar("T")

# Gemini Embedding 2 is steered with a text prefix. The same prefix must be used for every
# email, since emails are compared with each other.
EMBEDDING_TASK_PREFIX = "task: sentence similarity | query: "

# HTTP codes worth retrying: timeout, quota or rate limit, and server-side errors.
# Other 4xx codes (invalid key, unknown model, malformed request) fail the same way every time.
RETRYABLE_STATUS = {408, 429}


def is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS or status_code >= 500


def _thinking(thinking_level: str | None) -> types.ThinkingConfig | None:
    if not thinking_level:
        return None
    return types.ThinkingConfig(thinking_level=types.ThinkingLevel(thinking_level.upper()))


def build_config(
    *, system: str, schema: type[BaseModel], thinking_level: str | None
) -> types.GenerateContentConfig:
    """Request settings for one structured call.

    No temperature: Google asks not to set sampling parameters on Gemini 3 models (it can
    degrade answers). Answers can therefore vary from one run to the next, which is what
    the evaluation suite measures (step 3).
    """
    return types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=_thinking(thinking_level),
    )


def build_tools_config(
    *, system: str, tools: list[ToolSpec], thinking_level: str | None
) -> types.GenerateContentConfig:
    """Request settings for one step of the agent: the tools are declared, never run here."""
    declarations = [
        types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            # A tool without arguments is declared without a schema rather than with an
            # empty object, which some schema validators refuse.
            parameters_json_schema=tool.parameters if tool.parameters.get("properties") else None,
        )
        for tool in tools
    ]
    return types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=declarations)],
        # The SDK can run Python functions by itself; our tools are declarations only, and
        # this makes sure the agent loop is the one place where a tool runs.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=_thinking(thinking_level),
    )


def to_contents(messages: list[Message]) -> list[types.Content]:
    """Convert the conversation to Gemini's format."""
    contents: list[types.Content] = []
    for message in messages:
        if isinstance(message, UserMessage):
            contents.append(types.Content(role="user", parts=[types.Part(text=message.text)]))
        elif isinstance(message, ModelTurn):
            if isinstance(message.raw, types.Content):
                # Sent back exactly as received, with its thought signatures.
                contents.append(message.raw)
            else:
                parts = [types.Part(text=message.text)] if message.text else []
                parts += [
                    types.Part(
                        function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments)
                    )
                    for c in message.tool_calls
                ]
                contents.append(types.Content(role="model", parts=parts))
        elif isinstance(message, ToolResults):
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        id=r.call.id, name=r.call.name, response=r.content
                    )
                )
                for r in message.results
            ]
            contents.append(types.Content(role="user", parts=parts))
    return contents


def _token_counts(response: types.GenerateContentResponse) -> tuple[int, int]:
    usage = response.usage_metadata
    if usage is None:
        return 0, 0
    # "Thinking" tokens are billed as output tokens, so they count in the cost.
    output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return usage.prompt_token_count or 0, output


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_level: str | None = None,
        timeout_seconds: float = 90,
        embedding_model: str = "gemini-embedding-2",
        base_url: str | None = None,  # tests point the client at a local server
    ) -> None:
        self._client = genai.Client(
            api_key=api_key,
            # Milliseconds. The SDK sets no limit by default; it also does not retry
            # unless asked, so retries are only the worker's (one place, one policy).
            http_options=types.HttpOptions(
                timeout=round(timeout_seconds * 1000), base_url=base_url
            ),
        )
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_seconds = timeout_seconds
        self.embedding_model = embedding_model

    def _generate(
        self, contents: Any, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        return self._call(
            lambda: self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        )

    def _call(self, request: Callable[[], T]) -> T:
        """Run one API request and turn its failures into LLMError."""
        try:
            return request()
        except errors.APIError as exc:  # quota, invalid key, unknown model, server error...
            raise LLMError(f"Gemini API error: {exc}", retryable=is_retryable(exc.code)) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Gemini did not answer within {self.timeout_seconds:g} s", retryable=True
            ) from exc
        except httpx.HTTPError as exc:  # network failure, proxy refusal
            raise LLMError(f"Network error calling Gemini: {exc}", retryable=True) from exc

    def embed(self, text: str) -> list[float]:
        # Gemini Embedding 2 takes the task as a prefix instead of a task_type setting.
        # Emails are compared with emails, so both sides use the same "similarity" task.
        response = self._call(
            lambda: self._client.models.embed_content(
                model=self.embedding_model,
                contents=f"{EMBEDDING_TASK_PREFIX}{text}",
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
            )
        )
        values = response.embeddings[0].values if response.embeddings else None
        if not values or len(values) != EMBEDDING_DIMENSIONS:
            raise LLMError(f"Gemini returned an unexpected embedding ({len(values or [])} values)")
        return normalize(list(values))

    def generate_json(self, *, system: str, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        config = build_config(system=system, schema=schema, thinking_level=self.thinking_level)
        response = self._generate(prompt, config)
        input_tokens, output_tokens = _token_counts(response)
        return LLMResponse(
            text=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    def generate_with_tools(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelTurn:
        config = build_tools_config(system=system, tools=tools, thinking_level=self.thinking_level)
        response = self._generate(to_contents(messages), config)
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        if content is None or not content.parts:
            # For example MALFORMED_FUNCTION_CALL: the model produced an unusable answer.
            # Sampling is not fixed, so another attempt may well work.
            reason = candidate.finish_reason if candidate else "no candidate"
            raise LLMError(f"Gemini returned no content (finish reason: {reason})")

        text = "".join(p.text for p in content.parts if p.text and not p.thought)
        calls = [
            ToolCall(
                name=p.function_call.name or "",
                arguments=dict(p.function_call.args or {}),
                id=p.function_call.id,
            )
            for p in content.parts
            if p.function_call
        ]
        input_tokens, output_tokens = _token_counts(response)
        return ModelTurn(
            text=text.strip(),
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            raw=content,
        )
