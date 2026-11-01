from decimal import Decimal

from fastapi.testclient import TestClient

from app.api.deps import get_llm
from app.llm.fake import FakeLLMClient
from app.main import app
from tests.conftest import triage_json

PAYLOAD = {
    "sender": "camille.roux@example.com",
    "subject": "Vase cassé",
    "body": "Bonjour, ma commande ML-104201 est arrivée cassée.",
}


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_post_email_triages_and_logs_the_run(client: TestClient, fake_llm: FakeLLMClient) -> None:
    fake_llm.script = [triage_json()]

    response = client.post("/emails", json=PAYLOAD)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "triaged"
    assert body["triage"]["category"] == "damaged"
    assert body["triage"]["order_id"] == "ML-104201"
    run = body["runs"][0]
    assert run["status"] == "success"
    assert run["model"] == "fake-model"
    assert run["prompt_version"] == "triage-v1"
    assert run["input_tokens"] > 0
    assert Decimal(run["cost_usd"]) > 0


def test_email_with_flags_goes_to_review(client: TestClient, fake_llm: FakeLLMClient) -> None:
    fake_llm.script = [triage_json(urgency="high")]
    body = client.post("/emails", json=PAYLOAD).json()
    assert body["status"] == "needs_review"
    assert "high_urgency" in body["flags"]


def test_failed_triage_is_stored_with_the_raw_output(
    client: TestClient, fake_llm: FakeLLMClient
) -> None:
    fake_llm.script = ["oops", "oops again"]
    body = client.post("/emails", json=PAYLOAD).json()
    assert body["status"] == "failed"
    assert body["triage"] is None
    assert body["runs"][0]["status"] == "invalid_output"
    assert body["runs"][0]["attempts"] == 2


def test_list_can_filter_by_status(client: TestClient, fake_llm: FakeLLMClient) -> None:
    fake_llm.script = [triage_json(), triage_json(urgency="high")]
    client.post("/emails", json=PAYLOAD)
    client.post("/emails", json=PAYLOAD)

    review = client.get("/emails", params={"status": "needs_review"}).json()
    assert len(review) == 1
    assert len(client.get("/emails").json()) == 2


def test_get_unknown_email_returns_404(client: TestClient) -> None:
    assert client.get("/emails/999").status_code == 404


def test_rejects_empty_body(client: TestClient) -> None:
    response = client.post("/emails", json={**PAYLOAD, "body": ""})
    assert response.status_code == 422


def test_returns_503_when_no_model_is_configured(client: TestClient) -> None:
    # Restore the real dependency: no GEMINI_API_KEY is set in the test environment.
    del app.dependency_overrides[get_llm]
    response = client.post("/emails", json=PAYLOAD)
    assert response.status_code == 503
    assert client.get("/emails").json() == []
