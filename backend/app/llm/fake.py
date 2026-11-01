from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.llm.base import LLMError, LLMResponse

Script = list[str | Exception] | Callable[[str], str]


@dataclass
class FakeLLMClient:
    """Scripted stand-in for a real model, used in tests so they never call a paid API.

    `script` is either a list of answers returned in order (an Exception in the list is
    raised instead, to simulate a provider failure), or a function of the prompt.
    Every call is recorded so tests can check what was sent.
    """

    script: Script
    model: str = "fake-model"
    calls: list[dict[str, str]] = field(default_factory=list)

    def generate_json(self, *, system: str, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        self.calls.append({"system": system, "prompt": prompt, "schema": schema.__name__})
        if callable(self.script):
            text = self.script(prompt)
        else:
            if not self.script:
                raise LLMError("FakeLLMClient: no scripted answer left")
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            text = item
        # Rough token estimate (about 4 characters per token) so cost code is exercised.
        return LLMResponse(
            text=text,
            input_tokens=(len(system) + len(prompt)) // 4,
            output_tokens=len(text) // 4,
            model=self.model,
        )
