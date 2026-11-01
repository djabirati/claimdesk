from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.llm.base import LLMError, LLMResponse, Message, ModelTurn, ToolCall, ToolSpec

Script = list[str | Exception] | Callable[[str], str]
TurnScript = list[ModelTurn | Exception] | Callable[[list[Message]], ModelTurn] | None


def model_turn(*calls: tuple[str, dict[str, Any]], text: str = "") -> ModelTurn:
    """A scripted model answer: `model_turn(("get_order", {"order_id": "ML-104201"}))`."""
    return ModelTurn(
        text=text,
        tool_calls=[ToolCall(name=name, arguments=args) for name, args in calls],
        input_tokens=0,
        output_tokens=0,
        model="fake-model",
    )


@dataclass
class FakeLLMClient:
    """Scripted stand-in for a real model, used in tests so they never call a paid API.

    `script` drives generate_json: a list of answers returned in order (an Exception in
    the list is raised instead, to simulate a provider failure), or a function of the
    prompt. `turns` drives generate_with_tools the same way (a function receives the
    conversation so far). With no `turns`, the agent ends at once without proposing
    anything. Every call is recorded so tests can check what was sent.
    """

    script: Script
    turns: TurnScript = None
    model: str = "fake-model"
    calls: list[dict[str, Any]] = field(default_factory=list)

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

    def generate_with_tools(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelTurn:
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": [t.name for t in tools]}
        )
        if self.turns is None:
            turn = model_turn(text="Rien à proposer.")
        elif callable(self.turns):
            turn = self.turns(list(messages))
        else:
            if not self.turns:
                raise LLMError("FakeLLMClient: no scripted turn left")
            item = self.turns.pop(0)
            if isinstance(item, Exception):
                raise item
            turn = item
        size = len(system) + sum(len(repr(m)) for m in messages)
        return ModelTurn(
            text=turn.text,
            tool_calls=turn.tool_calls,
            input_tokens=size // 4,
            output_tokens=(len(turn.text) + len(repr(turn.tool_calls))) // 4,
            model=self.model,
        )
