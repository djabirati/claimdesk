import json
from collections.abc import Callable, Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401  (registers tables)
from app.config import Settings
from app.db import Base, get_session
from app.llm.fake import FakeLLMClient
from app.main import app
from app.models import Order
from app.worker import run_once

SEED_ORDERS = Path(__file__).resolve().parents[2] / "data" / "seed" / "orders.json"


def add_seed_orders(session: Session) -> None:
    """The 15 fictional orders of data/seed, as the agent's tools will find them."""
    for row in json.loads(SEED_ORDERS.read_text(encoding="utf-8")):
        session.add(
            Order(
                id=row["id"],
                customer_name=row["customer_name"],
                customer_email=row["customer_email"],
                total_amount=Decimal(row["total_amount"]),
                carrier=row["carrier"],
                tracking_number=row["tracking_number"],
                status=row["status"],
                shipped_at=datetime.fromisoformat(row["shipped_at"]),
                delivered_at=(
                    datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None
                ),
            )
        )
    session.commit()


def triage_json(**overrides: object) -> str:
    """A valid model answer; override only the fields a test cares about."""
    answer: dict[str, object] = {
        "category": "damaged",
        "urgency": "normal",
        "order_id": "ML-104201",
        "tracking_number": None,
        "requested_action": "replacement",
        "summary": "Le vase est arrivé cassé, la cliente demande un remplacement.",
        "confidence": 0.92,
    }
    answer.update(overrides)
    return json.dumps(answer)


@pytest.fixture
def settings() -> Settings:
    # Ignore any local .env file so tests behave the same everywhere.
    return Settings(_env_file=None, gemini_api_key=None)  # type: ignore[call-arg]


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    # In-memory SQLite shared across threads: fast, isolated, no server needed.
    # Migrations and the queue's locking are tested against real Postgres in CI.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(script=[])


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def override_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def drain(
    session_factory: sessionmaker[Session], fake_llm: FakeLLMClient, settings: Settings
) -> Callable[[], int]:
    """Run the worker until no job is due, as the background worker would. Returns how
    many jobs were taken. Retries scheduled in the future are left in the queue."""

    def _drain() -> int:
        taken = 0
        while run_once(session_factory, fake_llm, settings, worker_id="test-worker"):
            taken += 1
        return taken

    return _drain
