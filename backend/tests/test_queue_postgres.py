"""The queue and the worker on a real Postgres (SQLite ignores FOR UPDATE SKIP LOCKED
and has no view of open transactions).

Runs only when TEST_DATABASE_URL points to a disposable database already migrated with
`alembic upgrade head` (CI does this). Skipped otherwise.
"""

import os
import threading
from collections.abc import Iterator
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.enums import EmailStatus, JobKind, JobStatus
from app.jobs import claim_next_job, enqueue, utcnow
from app.llm.fake import FakeLLMClient
from app.models import Email, Job
from app.worker import run_once
from tests.conftest import triage_json

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")

TEST_SENDER = "queue-test@example.com"


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    with factory() as session:
        # Jobs are deleted with their emails (ON DELETE CASCADE).
        session.execute(delete(Email).where(Email.sender == TEST_SENDER))
        session.commit()
    engine.dispose()


def add_jobs(factory: sessionmaker[Session], count: int) -> list[int]:
    """Queue `count` jobs, oldest first, and return their ids in that order."""
    start = utcnow() - timedelta(minutes=1)
    with factory() as session:
        jobs = []
        for i in range(count):
            email = Email(
                sender=TEST_SENDER, subject=f"Test {i}", body="...", status=EmailStatus.RECEIVED
            )
            session.add(email)
            jobs.append(
                enqueue(
                    session,
                    email=email,
                    kind=JobKind.TRIAGE,
                    max_attempts=4,
                    now=start + timedelta(milliseconds=i),
                )
            )
        session.commit()
        return [job.id for job in jobs]


def test_a_locked_job_is_skipped_instead_of_waited_for(factory: sessionmaker[Session]) -> None:
    first, second = add_jobs(factory, 2)
    with factory() as other_worker, factory() as worker:
        # Another worker is in the middle of claiming the first job: its row is locked.
        other_worker.execute(text("SELECT id FROM jobs WHERE id = :id FOR UPDATE"), {"id": first})
        # Without SKIP LOCKED the claim below would wait for that lock; fail fast instead.
        worker.execute(text("SET lock_timeout = '2s'"))

        job = claim_next_job(worker, worker_id="worker-2")

        assert job is not None
        assert job.id == second
        other_worker.rollback()


def test_concurrent_workers_never_take_the_same_job(factory: sessionmaker[Session]) -> None:
    job_ids = add_jobs(factory, 40)
    claimed: list[int] = []
    guard = threading.Lock()

    def worker(number: int) -> None:
        with factory() as session:
            while (job := claim_next_job(session, worker_id=f"worker-{number}")) is not None:
                with guard:
                    claimed.append(job.id)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Every job was taken exactly once.
    assert sorted(claimed) == sorted(job_ids)


def test_no_transaction_stays_open_during_the_model_call(factory: sessionmaker[Session]) -> None:
    # A model call can take tens of seconds. A transaction left open meanwhile keeps a
    # Postgres connection "idle in transaction" (it holds a snapshot and blocks clean-up).
    (job_id,) = add_jobs(factory, 1)
    assert DATABASE_URL is not None
    probe = create_engine(DATABASE_URL)
    idle_in_transaction: list[int] = []

    def answer(prompt: str) -> str:
        # Runs while the worker waits for the "model": look at the server's connections.
        with probe.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND state = 'idle in transaction'"
                )
            ).scalar_one()
        idle_in_transaction.append(count)
        return triage_json()

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert run_once(factory, FakeLLMClient(script=answer), settings, worker_id="worker-1")
    probe.dispose()

    assert idle_in_transaction == [0]
    with factory() as session:
        assert session.get_one(Job, job_id).status == JobStatus.DONE
