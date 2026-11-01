import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401  (registers tables)
from app.api.deps import get_llm
from app.db import Base, get_session
from app.llm.fake import FakeLLMClient
from app.main import app


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
def session_factory() -> Iterator[sessionmaker[Session]]:
    # In-memory SQLite shared across threads: fast, isolated, no server needed.
    # Migrations themselves are tested against real Postgres in CI.
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
def client(session_factory: sessionmaker[Session], fake_llm: FakeLLMClient) -> Iterator[TestClient]:
    def override_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_llm] = lambda: fake_llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
