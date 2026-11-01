from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


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


class LLMClient(Protocol):
    """What the rest of the code needs from a model provider.

    Keeping this interface small means the triage logic never imports a provider SDK:
    switching from Gemini to another model only means writing a new client.
    """

    model: str

    def generate_json(self, *, system: str, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        """Ask for a JSON answer that should follow `schema`. Validation is the caller's job."""
        ...
