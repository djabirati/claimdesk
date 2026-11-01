"""Integration test against a real Postgres (JSONB, Numeric, timezones).

Runs only when TEST_DATABASE_URL points to a database already migrated with
`alembic upgrade head` (CI does this). Skipped otherwise.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.llm.fake import FakeLLMClient
from app.models import Email
from app.triage.pipeline import triage_email
from tests.conftest import triage_json

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")


def test_triage_round_trip_on_postgres() -> None:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        email = Email(
            sender="camille.roux@example.com",
            subject="Vase cassé",
            body="Ma commande ML-104201 est cassée.",
        )
        session.add(email)
        session.commit()

        run = triage_email(session, email, FakeLLMClient(script=[triage_json()]), Settings())

        assert run.id is not None
        assert run.created_at.tzinfo is not None
        stored = session.execute(
            text("SELECT triage->>'order_id' FROM emails WHERE id = :id"), {"id": email.id}
        ).scalar_one()
        assert stored == "ML-104201"

        session.delete(email)
        session.commit()
    engine.dispose()
