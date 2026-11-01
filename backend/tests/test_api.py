from collections.abc import Callable
from decimal import Decimal

from fastapi.testclient import TestClient

from app.llm.fake import FakeLLMClient
from tests.conftest import triage_json

PAYLOAD = {
    "sender": "camille.roux@example.com",
    "subject": "Vase cassé",
    "body": "Bonjour, ma commande ML-104201 est arrivée cassée.",
}

Drain = Callable[[], int]


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_post_email_queues_it_without_calling_the_model(
    client: TestClient, fake_llm: FakeLLMClient
) -> None:
    response = client.post("/emails", json=PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "received"
    assert body["runs"] == []
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["kind"] == "triage"
    assert body["jobs"][0]["status"] == "queued"
    assert fake_llm.calls == []


def test_worker_triages_the_email_and_logs_the_run(
    client: TestClient, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json()]
    email_id = client.post("/emails", json=PAYLOAD).json()["id"]

    assert drain() == 1

    body = client.get(f"/emails/{email_id}").json()
    assert body["status"] == "triaged"
    assert body["triage"]["category"] == "damaged"
    assert body["triage"]["order_id"] == "ML-104201"
    run = body["runs"][0]
    assert run["status"] == "success"
    assert run["model"] == "fake-model"
    assert run["prompt_version"] == "triage-v1"
    assert run["input_tokens"] > 0
    assert Decimal(run["cost_usd"]) > 0
    job = body["jobs"][0]
    assert job["status"] == "done"
    assert job["attempts"] == 1
    assert job["finished_at"] is not None


def test_email_with_flags_goes_to_review(
    client: TestClient, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json(urgency="high")]
    email_id = client.post("/emails", json=PAYLOAD).json()["id"]
    drain()

    body = client.get(f"/emails/{email_id}").json()
    assert body["status"] == "needs_review"
    assert "high_urgency" in body["flags"]


def test_invalid_output_fails_the_email_without_a_job_retry(
    client: TestClient, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = ["oops", "oops again"]
    email_id = client.post("/emails", json=PAYLOAD).json()["id"]
    drain()

    body = client.get(f"/emails/{email_id}").json()
    assert body["status"] == "failed"
    assert body["triage"] is None
    assert body["runs"][0]["status"] == "invalid_output"
    assert body["runs"][0]["attempts"] == 2
    assert body["jobs"][0]["status"] == "failed"
    assert body["jobs"][0]["attempts"] == 1
    assert len(fake_llm.calls) == 2  # the retry inside classify(), nothing more


def test_list_can_filter_by_status(
    client: TestClient, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json(), triage_json(urgency="high")]
    client.post("/emails", json=PAYLOAD)
    client.post("/emails", json=PAYLOAD)
    assert len(client.get("/emails", params={"status": "received"}).json()) == 2

    drain()

    assert len(client.get("/emails", params={"status": "needs_review"}).json()) == 1
    assert len(client.get("/emails", params={"status": "received"}).json()) == 0
    assert len(client.get("/emails").json()) == 2


def test_get_unknown_email_returns_404(client: TestClient) -> None:
    assert client.get("/emails/999").status_code == 404


def test_rejects_empty_body(client: TestClient) -> None:
    response = client.post("/emails", json={**PAYLOAD, "body": ""})
    assert response.status_code == 422
