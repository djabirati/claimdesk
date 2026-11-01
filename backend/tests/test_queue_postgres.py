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

from app.actions import ActionNotPending, approve_action
from app.agent.loop import run_agent
from app.config import Settings
from app.enums import ActionKind, EmailStatus, JobKind, JobStatus
from app.jobs import claim_next_job, enqueue, utcnow
from app.llm.base import Message, ModelTurn
from app.llm.fake import FakeLLMClient, model_turn
from app.models import AgentRun, Email, Job, ProposedAction
from app.worker import run_once
from tests.conftest import triage_json

IDLE_IN_TRANSACTION = text(
    "SELECT count(*) FROM pg_stat_activity "
    "WHERE datname = current_database() AND state = 'idle in transaction'"
)

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
            idle_in_transaction.append(conn.execute(IDLE_IN_TRANSACTION).scalar_one())
        return triage_json()

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert run_once(factory, FakeLLMClient(script=answer), settings, worker_id="worker-1")
    probe.dispose()

    assert idle_in_transaction == [0]
    with factory() as session:
        assert session.get_one(Job, job_id).status == JobStatus.DONE


def triaged_claim(factory: sessionmaker[Session]) -> int:
    with factory() as session:
        email = Email(
            sender=TEST_SENDER,
            subject="Vase cassé",
            body="Cassé.",
            status=EmailStatus.TRIAGED,
            triage={"category": "damaged"},
            flags=[],
        )
        session.add(email)
        session.commit()
        return email.id


def test_no_transaction_stays_open_during_the_agent_calls(
    factory: sessionmaker[Session],
) -> None:
    email_id = triaged_claim(factory)
    assert DATABASE_URL is not None
    probe = create_engine(DATABASE_URL)
    idle_in_transaction: list[int] = []
    script = [
        model_turn(("list_customer_orders", {})),  # the tool reads the database
        model_turn(text="Aucune commande trouvée."),
    ]

    def turn(messages: list[Message]) -> ModelTurn:
        with probe.connect() as conn:
            idle_in_transaction.append(conn.execute(IDLE_IN_TRANSACTION).scalar_one())
        return script.pop(0)

    with factory() as session:
        email = session.get_one(Email, email_id)
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        run_agent(session, email, FakeLLMClient(script=[], turns=turn), settings)
    probe.dispose()

    assert idle_in_transaction == [0, 0]  # before each model call, after the tool's read


def test_two_people_approving_at_once_execute_the_refund_once(
    factory: sessionmaker[Session],
) -> None:
    email_id = triaged_claim(factory)
    with factory() as session:
        run = AgentRun(
            email_id=email_id,
            model="fake",
            prompt_version="test",
            status="completed",
            steps=1,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            cost_usd=0,
        )
        action = ProposedAction(
            email_id=email_id,
            agent_run=run,
            kind=ActionKind.REFUND,
            order_id="ML-104201",
            payload={"amount_eur": "89.90"},
        )
        session.add_all([run, action])
        session.commit()
        action_id = action.id

    start = threading.Barrier(2)
    outcomes: list[str] = []

    def reviewer(name: str) -> None:
        with factory() as session:
            start.wait()  # both click "approve" at the same moment
            try:
                approve_action(session, action_id, reviewer=name)
                outcomes.append("executed")
            except ActionNotPending:
                outcomes.append("already decided")

    threads = [threading.Thread(target=reviewer, args=(name,)) for name in ("Djabi", "Sam")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["already decided", "executed"]
    with factory() as session:
        assert session.get_one(ProposedAction, action_id).status == "executed"
