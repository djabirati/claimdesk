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
from pydantic import ValidationError

from app.config import Settings
from app.llm.base import LLMError
from app.llm.gemini import GeminiClient, is_retryable
from app.schemas import TriageOutput


class FakeGemini:
    """A local HTTP server that answers like the Gemini API."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.delay_seconds = 0.0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                fake.requests.append(
                    {"path": self.path, "body": json.loads(self.rfile.read(length))}
                )
                time.sleep(fake.delay_seconds)
                answer = {
                    "candidates": [{"content": {"role": "model", "parts": [{"text": "{}"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 120,
                        "candidatesTokenCount": 40,
                        "thoughtsTokenCount": 15,
                    },
                }
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
