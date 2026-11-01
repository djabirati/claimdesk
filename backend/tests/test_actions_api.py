"""The approval queue: list what the agent proposed, approve or reject each proposal."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.llm.fake import FakeLLMClient, model_turn
from tests.conftest import add_seed_orders, triage_json

PAYLOAD = {
    "sender": "camille.roux@example.com",
    "subject": "Vase cassé",
    "body": "Bonjour, ma commande ML-104201 est arrivée cassée. Je voudrais être remboursée.",
}
REPLY = "Bonjour, nous avons bien reçu votre demande et notre équipe s'en occupe."
REVIEWER = {"reviewer": "Djabi", "note": "Vérifié avec la photo."}


@pytest.fixture
def proposals(
    client: TestClient,
    session_factory: sessionmaker[Session],
    fake_llm: FakeLLMClient,
    drain: Callable[[], int],
) -> dict[str, int]:
    """An email triaged and seen by the agent, which proposed a refund and a reply."""
    with session_factory() as session:
        add_seed_orders(session)
    fake_llm.script = [triage_json(requested_action="refund")]
    fake_llm.turns = [
        model_turn(
            ("propose_refund", {"order_id": "ML-104201", "amount_eur": 89.9, "reason": "Cassé."}),
            ("propose_reply", {"message": REPLY}),
        ),
        model_turn(text="Commande vérifiée, je propose le remboursement complet."),
    ]
    client.post("/emails", json=PAYLOAD)
    drain()
    queue = client.get("/actions", params={"status": "proposed"}).json()
    return {action["kind"]: action["id"] for action in queue}


def test_the_queue_lists_what_the_agent_proposed(
    client: TestClient, proposals: dict[str, int]
) -> None:
    queue = client.get("/actions", params={"status": "proposed"}).json()

    assert [a["kind"] for a in queue] == ["refund", "reply"]
    refund = queue[0]
    assert refund["payload"] == {"amount_eur": "89.90"}
    assert refund["order_id"] == "ML-104201"
    assert refund["rationale"] == "Cassé."
    assert refund["executed_at"] is None


def test_approving_executes_the_action_once(client: TestClient, proposals: dict[str, int]) -> None:
    refund_id = proposals["refund"]

    response = client.post(f"/actions/{refund_id}/approve", json=REVIEWER)

    assert response.status_code == 200
    action = response.json()
    assert action["status"] == "executed"
    assert action["decided_by"] == "Djabi"
    assert action["decision_note"] == "Vérifié avec la photo."
    assert action["result"] == {"refund_reference": f"RF-{refund_id:06d}", "amount_eur": "89.90"}
    # A second click, or a second person, cannot pay twice.
    again = client.post(f"/actions/{refund_id}/approve", json=REVIEWER)
    assert again.status_code == 409
    assert client.post(f"/actions/{refund_id}/reject", json=REVIEWER).status_code == 409


def test_a_rejected_action_never_runs(client: TestClient, proposals: dict[str, int]) -> None:
    reply_id = proposals["reply"]

    action = client.post(f"/actions/{reply_id}/reject", json={"reviewer": "Djabi"}).json()

    assert action["status"] == "rejected"
    assert action["executed_at"] is None
    assert action["result"] is None
    assert client.post(f"/actions/{reply_id}/approve", json=REVIEWER).status_code == 409
    assert [a["kind"] for a in client.get("/actions?status=proposed").json()] == ["refund"]


def test_the_email_shows_the_agent_run_and_its_proposals(
    client: TestClient, proposals: dict[str, int]
) -> None:
    email = client.get("/emails").json()[0]
    detail = client.get(f"/emails/{email['id']}").json()

    (agent_run,) = detail["agent_runs"]
    assert agent_run["status"] == "completed"
    assert agent_run["note"] == "Commande vérifiée, je propose le remboursement complet."
    assert [entry["tool"] for entry in agent_run["trace"]] == ["propose_refund", "propose_reply"]
    assert [a["kind"] for a in detail["actions"]] == ["refund", "reply"]


def test_unknown_action_returns_404(client: TestClient) -> None:
    assert client.get("/actions/999").status_code == 404
    assert client.post("/actions/999/approve", json=REVIEWER).status_code == 404


def test_a_decision_needs_a_reviewer(client: TestClient, proposals: dict[str, int]) -> None:
    response = client.post(f"/actions/{proposals['refund']}/approve", json={})
    assert response.status_code == 422
