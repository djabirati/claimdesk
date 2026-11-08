from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


# ---------- Tool calling (used by the agent) ----------
# Provider-neutral types: the agent loop never sees a provider SDK object, except as the
# opaque `raw` field that the provider client gives back to itself on the next call.


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: its name, what it does, and a JSON schema of its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None  # set by providers that number their calls


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    content: dict[str, Any]


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class ModelTurn:
    """One answer of the model: some text, some tool calls, or both."""

    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    model: str
    # The provider's own message, sent back unchanged in the next request. Gemini 3 needs
    # this: its answers carry "thought signatures" that must be returned exactly.
    raw: Any = None


@dataclass(frozen=True)
class ToolResults:
    results: list[ToolResult] = field(default_factory=list)


Message = UserMessage | ModelTurn | ToolResults


class LLMError(Exception):
    """The provider gave no answer. Not a bad answer: no answer at all.

    `retryable` says whether the same call could work later. True for a network failure,
    a quota or a server error; False when the provider rejects the request itself
    (invalid key, unknown model, malformed request), which a retry would not fix.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMNotConfiguredError(Exception):
    """No provider credentials were given."""


# Fixed by the database column (migration 0004): changing it means a new migration and
# embedding every email again.
EMBEDDING_DIMENSIONS = 768


def normalize(vector: list[float]) -> list[float]:
    """Scale to length 1. A vector cut to fewer dimensions is no longer unit-length."""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector


class LLMClient(Protocol):
    """What the rest of the code needs from a model provider.

    Keeping this interface small means the triage and agent logic never import a provider
    SDK: switching from Gemini to another model only means writing a new client.
    """

    model: str
    embedding_model: str

    def generate_json(self, *, system: str, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        """Ask for a JSON answer that should follow `schema`. Validation is the caller's job."""
        ...

    def generate_with_tools(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelTurn:
        """Continue a conversation in which the model may call tools.

        The client never runs a tool itself: it returns the calls, and the caller decides
        what to do with them.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """A unit-length vector of EMBEDDING_DIMENSIONS numbers, for similarity search.

        Texts compared with each other must be embedded by the same model.
        """
        ...
