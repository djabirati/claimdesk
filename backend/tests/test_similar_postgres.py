"""Similar past cases on a real Postgres with pgvector (SQLite has no vector search).

Runs only when TEST_DATABASE_URL points to a disposable database already migrated with
`alembic upgrade head` (CI does this). Skipped otherwise.
"""

import json
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session, sessionmaker

from app.agent.loop import run_agent
from app.config import Settings
from app.enums import ActionKind, ActionStatus, EmailStatus
from app.llm.base import EMBEDDING_DIMENSIONS, normalize
from app.llm.fake import FakeLLMClient, model_turn
from app.models import AgentRun, Email, EmailEmbedding, ProposedAction
from app.similar import find_similar_cases

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")

DOMAIN = "@similar.test"


@pytest.fixture
def session() -> Iterator[Session]:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        yield session
        session.rollback()
        session.execute(delete(Email).where(Email.sender.like(f"%{DOMAIN}")))
        session.commit()
    engine.dispose()


def vector(*head: float) -> list[float]:
    """A unit vector whose first values are `head`, the rest zeros."""
    values = list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))
    return normalize(values)


def past_case(
    session: Session,
    name: str,
    embedding: list[float],
    decisions: list[tuple[ActionKind, ActionStatus]],
    *,
    model: str = "fake-embedding",
    category: str = "damaged",
) -> Email:
    email = Email(
        sender=f"{name}{DOMAIN}",
        subject=f"Réclamation de {name}",
        body=f"Bonjour, je suis {name}, mon colis est arrivé abîmé.",
        status=EmailStatus.TRIAGED,
        triage={"category": category, "requested_action": "replacement"},
        flags=[],
    )
    run = AgentRun(
        email=email,
        model="fake",
        prompt_version="test",
        status="completed",
        steps=1,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        cost_usd=0,
        note=f"Note de l'agent sur {name}",
        flags=[],
        trace=[],
    )
    session.add_all([email, run, EmailEmbedding(email=email, model=model, embedding=embedding)])
    for kind, status in decisions:
        session.add(
            ProposedAction(
                email=email,
                agent_run=run,
                kind=kind,
                status=status,
                order_id="ML-999999",
                payload={"amount_eur": "40.00"} if kind is ActionKind.REFUND else {},
                rationale=f"Raison écrite pour {name}",
                decision_note=f"Note du relecteur sur {name}",
            )
        )
    session.commit()
    return email


def test_returns_the_closest_cases_decided_by_a_person(session: Session) -> None:
    current = past_case(session, "current", vector(1, 0), [])
    past_case(
        session,
        "close",
        vector(0.95, 0.31),
        [
            (ActionKind.REPLACEMENT, ActionStatus.EXECUTED),
            (ActionKind.REFUND, ActionStatus.REJECTED),
        ],
    )
    past_case(session, "closer", vector(0.99, 0.14), [(ActionKind.REFUND, ActionStatus.EXECUTED)])
    # Excluded: no decision yet, another embedding model, too far away.
    past_case(session, "undecided", vector(1, 0.01), [(ActionKind.REFUND, ActionStatus.PROPOSED)])
    past_case(
        session,
        "othermodel",
        vector(1, 0),
        [(ActionKind.REFUND, ActionStatus.EXECUTED)],
        model="other",
    )
    past_case(session, "far", vector(0, 1), [(ActionKind.REPLY, ActionStatus.EXECUTED)])

    cases = find_similar_cases(session, current, limit=5, min_similarity=0.6)

    assert [c["similarity"] for c in cases] == [0.99, 0.951]
    assert cases[0]["decisions"] == [
        {"action": "refund", "decision": "executed", "amount_eur": "40.00"}
    ]
    assert cases[1]["decisions"] == [
        {"action": "replacement", "decision": "executed"},
        {"action": "refund", "decision": "rejected", "amount_eur": "40.00"},
    ]


def test_a_past_case_never_carries_another_customers_text(session: Session) -> None:
    current = past_case(session, "current", vector(1, 0), [])
    past_case(session, "alice", vector(1, 0.05), [(ActionKind.REFUND, ActionStatus.EXECUTED)])

    (case,) = find_similar_cases(session, current, limit=3, min_similarity=0.6)

    assert set(case) == {"similarity", "category", "requested_action", "decisions"}
    as_text = json.dumps(case, ensure_ascii=False)
    # Nothing written by or about the other customer: no name, address, text, notes.
    for leaked in ("alice", "@", "abîmé", "Note", "Raison", "ML-999999"):
        assert leaked not in as_text


def test_the_agent_embeds_the_email_and_finds_similar_cases(session: Session) -> None:
    text = "Vase cassé à la livraison, je voudrais un remboursement du vase cassé."
    past = past_case(
        session, "past", vector(1, 0), [(ActionKind.REPLACEMENT, ActionStatus.EXECUTED)]
    )
    llm = FakeLLMClient(script=[])
    # Same text for both emails: the bag-of-words embedding gives them similarity 1.
    session.get_one(EmailEmbedding, past.id).embedding = llm.embedder(text)  # type: ignore[operator]
    current = Email(
        sender=f"new{DOMAIN}",
        subject="",
        body=text,
        status=EmailStatus.TRIAGED,
        triage={"category": "damaged"},
        flags=[],
    )
    session.add(current)
    session.commit()
    llm.turns = [model_turn(("find_similar_cases", {})), model_turn(text="Fini.")]

    agent_run = run_agent(session, current, llm, Settings(_env_file=None))  # type: ignore[call-arg]

    assert session.get(EmailEmbedding, current.id) is not None
    (entry,) = agent_run.trace
    assert entry["result"]["cases"][0]["decisions"] == [
        {"action": "replacement", "decision": "executed"}
    ]
    assert entry["result"]["cases"][0]["similarity"] == 1.0
