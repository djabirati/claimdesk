"""The Gemini client, checked against a local stand-in server instead of the real API.

The server records what the client sends and can answer late, so the request settings
and the timeout are tested for real, for free, in CI.
"""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from google.genai import types
from pydantic import ValidationError

from app.config import Settings
from app.llm.base import LLMError, ToolResult, ToolResults, ToolSpec, UserMessage
from app.llm.gemini import GeminiClient, is_retryable
from app.schemas import TriageOutput


class FakeGemini:
    """A local HTTP server that answers like the Gemini API."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.delay_seconds = 0.0
        # Candidates to answer with, in order; the default answer once they run out.
        self.candidates: list[dict[str, Any]] = []
        # Values returned by the embedding endpoint.
        self.embedding: list[float] = [3.0, 4.0] + [0.0] * 766
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                fake.requests.append(
                    {"path": self.path, "body": json.loads(self.rfile.read(length))}
                )
                time.sleep(fake.delay_seconds)
                if "embedcontent" in self.path.lower():  # :batchEmbedContents
                    self._send({"embeddings": [{"values": fake.embedding}]})
                    return
                default = {"content": {"role": "model", "parts": [{"text": "{}"}]}}
                answer = {
                    "candidates": [fake.candidates.pop(0) if fake.candidates else default],
                    "usageMetadata": {
                        "promptTokenCount": 120,
                        "candidatesTokenCount": 40,
                        "thoughtsTokenCount": 15,
                    },
                }
                self._send(answer)

            def _send(self, answer: dict[str, Any]) -> None:
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request: object, client_address: object) -> None:
                pass  # the client hung up after a timeout: expected in the timeout test

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGemini]:
    # A proxy set in the environment must not intercept calls to the local server.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    fake = FakeGemini()
    thread = threading.Thread(target=fake.server.serve_forever, daemon=True)
    thread.start()
    yield fake
    fake.server.shutdown()


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


def test_request_asks_for_json_with_low_thinking_and_no_temperature(
    fake_gemini: FakeGemini,
) -> None:
    client = GeminiClient(
        api_key="test", model="gemini-3.8-flash", thinking_level="low", base_url=fake_gemini.url
    )

    response = client.generate_json(system="sys", prompt="hello", schema=TriageOutput)

    (request,) = fake_gemini.requests
    assert request["path"].endswith("/models/gemini-3.8-flash:generateContent")
    config = request["body"]["generationConfig"]
    assert "temperature" not in config  # Google asks not to set it on Gemini 3 models
    assert config["responseMimeType"] == "application/json"
    assert "damaged" in config["responseSchema"]["properties"]["category"]["enum"]
    assert config["thinkingConfig"] == {"thinking_level": "LOW"}
    # Thinking tokens are billed as output, so they are counted in the output tokens.
    assert (response.input_tokens, response.output_tokens) == (120, 55)


def test_no_thinking_setting_is_sent_when_none_is_configured(fake_gemini: FakeGemini) -> None:
    client = GeminiClient(api_key="test", model="gemini-2.5-flash", base_url=fake_gemini.url)
    client.generate_json(system="sys", prompt="hello", schema=TriageOutput)
    assert "thinkingConfig" not in fake_gemini.requests[0]["body"]["generationConfig"]


def test_a_call_that_takes_too_long_raises_a_retryable_error(fake_gemini: FakeGemini) -> None:
    fake_gemini.delay_seconds = 2
    client = GeminiClient(
        api_key="test", model="gemini-3.8-flash", timeout_seconds=0.3, base_url=fake_gemini.url
    )

    start = time.perf_counter()
    with pytest.raises(LLMError) as raised:
        client.generate_json(system="sys", prompt="hello", schema=TriageOutput)

    assert time.perf_counter() - start < 1.5  # gave up at the timeout, did not wait 2 s
    assert raised.value.retryable
    assert "did not answer within 0.3 s" in str(raised.value)


def test_empty_thinking_level_in_env_means_none() -> None:
    settings = Settings(_env_file=None, llm_thinking_level="")  # type: ignore[call-arg, arg-type]
    assert settings.llm_thinking_level is None


def test_settings_refuse_a_triage_longer_than_the_lock_timeout() -> None:
    # 2 calls x 200 s could outlast the 300 s lock: the job would go to a second worker.
    with pytest.raises(ValidationError, match="must be below job_lock_timeout_seconds"):
        Settings(_env_file=None, llm_timeout_seconds=200)  # type: ignore[call-arg]


# ---------- tool calling (the agent) ----------

GET_ORDER = ToolSpec(
    name="get_order",
    description="Details of one order.",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)
SIGNATURE = "c2lnbmF0dXJlLWZyb20tZ2VtaW5p"  # base64, as the API sends it


def test_tool_calls_and_thought_signatures_go_back_to_gemini_unchanged(
    fake_gemini: FakeGemini,
) -> None:
    # Gemini 3 rejects (400) a conversation whose function calls lost their thought
    # signature, so the model's turn must be sent back exactly as it came.
    fake_gemini.candidates = [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "call-1",
                            "name": "get_order",
                            "args": {"order_id": "ML-104201"},
                        },
                        "thoughtSignature": SIGNATURE,
                    }
                ],
            }
        },
        {"content": {"role": "model", "parts": [{"text": "Commande vérifiée."}]}},
    ]
    client = GeminiClient(
        api_key="test", model="gemini-3.8-flash", thinking_level="low", base_url=fake_gemini.url
    )
    messages: list[Any] = [UserMessage("Le vase est cassé.")]

    turn = client.generate_with_tools(system="sys", messages=messages, tools=[GET_ORDER])
    (call,) = turn.tool_calls
    assert (call.id, call.name, call.arguments) == (
        "call-1",
        "get_order",
        {"order_id": "ML-104201"},
    )

    messages += [turn, ToolResults([ToolResult(call=call, content={"total_eur": "89.90"})])]
    final = client.generate_with_tools(system="sys", messages=messages, tools=[GET_ORDER])
    assert final.text == "Commande vérifiée."
    assert final.tool_calls == []

    first, second = (request["body"] for request in fake_gemini.requests)
    assert first["tools"][0]["functionDeclarations"][0]["name"] == "get_order"
    user, model, tool = second["contents"]
    assert model["role"] == "model"
    assert model["parts"][0]["thoughtSignature"] == SIGNATURE
    assert model["parts"][0]["functionCall"]["id"] == "call-1"
    assert tool["parts"][0]["functionResponse"] == {
        "id": "call-1",
        "name": "get_order",
        "response": {"total_eur": "89.90"},
    }


def test_an_empty_answer_is_a_retryable_error(fake_gemini: FakeGemini) -> None:
    fake_gemini.candidates = [{"finishReason": "MALFORMED_FUNCTION_CALL"}]
    client = GeminiClient(api_key="test", model="gemini-3.8-flash", base_url=fake_gemini.url)

    with pytest.raises(LLMError) as raised:
        client.generate_with_tools(system="sys", messages=[UserMessage("x")], tools=[GET_ORDER])

    assert raised.value.retryable
    assert "MALFORMED_FUNCTION_CALL" in str(raised.value)


def test_every_agent_tool_is_declared_and_no_arg_tools_have_no_schema() -> None:
    from app.agent.tools import TOOL_SPECS
    from app.llm.gemini import build_tools_config

    config = build_tools_config(system="sys", tools=TOOL_SPECS, thinking_level="low")

    assert config.tools is not None
    tool = config.tools[0]
    assert isinstance(tool, types.Tool)
    declarations = tool.function_declarations or []
    by_name = {d.name: d for d in declarations}
    assert set(by_name) == {spec.name for spec in TOOL_SPECS}
    assert by_name["list_customer_orders"].parameters_json_schema is None
    assert by_name["propose_refund"].parameters_json_schema["required"] == [  # type: ignore[index]
        "order_id",
        "amount_eur",
        "reason",
    ]
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True


# ---------- embeddings (similar cases) ----------


def test_embeddings_use_the_similarity_prefix_and_768_normalised_values(
    fake_gemini: FakeGemini,
) -> None:
    client = GeminiClient(api_key="test", model="gemini-3.8-flash", base_url=fake_gemini.url)

    vector = client.embed("Vase cassé")

    (request,) = fake_gemini.requests
    assert "gemini-embedding-2" in request["path"]
    sent = json.dumps(request["body"], ensure_ascii=False)
    assert "task: sentence similarity | query: Vase cassé" in sent
    assert '"outputDimensionality": 768' in sent
    assert vector[:2] == pytest.approx([0.6, 0.8])  # 3-4-5 triangle: scaled to length 1
    assert len(vector) == 768


def test_an_embedding_of_the_wrong_size_is_an_error(fake_gemini: FakeGemini) -> None:
    fake_gemini.embedding = [1.0, 2.0]
    client = GeminiClient(api_key="test", model="gemini-3.8-flash", base_url=fake_gemini.url)

    with pytest.raises(LLMError, match="unexpected embedding"):
        client.embed("Vase cassé")
