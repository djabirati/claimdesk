"""The worker's decisions: what is retried, when it gives up, how abandoned jobs come back.

These run on SQLite, which ignores row locks. The locking itself is tested against
Postgres in test_queue_postgres.py.
"""

from collections.abc import Callable
from datetime import timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.enums import EmailStatus, JobKind, JobStatus
from app.jobs import claim_next_job, enqueue, requeue_stale_jobs, retry_delay, utcnow
from app.llm.base import LLMError
from app.llm.fake import FakeLLMClient
from app.models import Email, Job
from app.worker import main
from tests.conftest import triage_json

Drain = Callable[[], int]
Factory = sessionmaker[Session]


def add_email(session_factory: Factory, *, max_attempts: int = 4) -> int:
    with session_factory() as session:
        email = Email(
            sender="camille.roux@example.com",
            subject="Vase cassé",
            body="Ma commande ML-104201 est arrivée cassée.",
            status=EmailStatus.RECEIVED,
        )
        session.add(email)
        enqueue(session, email=email, kind=JobKind.TRIAGE, max_attempts=max_attempts)
        session.commit()
        return email.id


def load(session_factory: Factory, email_id: int) -> tuple[Email, Job]:
    with session_factory() as session:
        email = session.get_one(Email, email_id)
        job = email.jobs[0]
        _ = email.runs  # load the runs before the session closes
        return email, job


def make_retries_due(session_factory: Factory) -> None:
    """Skip the waiting time of scheduled retries."""
    with session_factory() as session:
        session.execute(update(Job).values(run_after=utcnow() - timedelta(seconds=1)))
        session.commit()


def claim_and_die(session_factory: Factory) -> None:
    """A worker claimed the job ten minutes ago, then died without finishing it."""
    with session_factory() as session:
        claim_next_job(session, worker_id="dead-worker")
        session.execute(update(Job).values(locked_at=utcnow() - timedelta(minutes=10)))
        session.commit()


def test_retry_delay_grows_four_times_each_failure() -> None:
    assert [retry_delay(n, 15).total_seconds() for n in (1, 2, 3)] == [15, 60, 240]


def test_provider_error_is_retried_later_then_succeeds(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [LLMError("429 quota exceeded"), triage_json()]
    email_id = add_email(session_factory)

    assert drain() == 1  # first attempt fails; the retry is not due yet
    email, job = load(session_factory, email_id)
    assert job.status == JobStatus.QUEUED
    assert job.attempts == 1
    assert "quota" in (job.last_error or "")
    assert email.status == EmailStatus.RECEIVED  # not failed: another attempt is planned

    make_retries_due(session_factory)
    assert drain() == 1

    email, job = load(session_factory, email_id)
    assert email.status == EmailStatus.TRIAGED
    assert job.status == JobStatus.DONE
    assert job.attempts == 2
    assert [run.status for run in email.runs] == ["error", "success"]  # both runs are logged


def test_job_fails_when_no_attempt_is_left(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [LLMError("server error"), LLMError("server error again")]
    email_id = add_email(session_factory, max_attempts=2)

    drain()
    make_retries_due(session_factory)
    drain()

    email, job = load(session_factory, email_id)
    assert job.status == JobStatus.FAILED
    assert job.attempts == 2
    assert "again" in (job.last_error or "")
    assert job.finished_at is not None
    assert email.status == EmailStatus.FAILED


def test_unexpected_error_keeps_the_worker_alive_and_retries(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("boom")

    fake_llm.script = broken
    email_id = add_email(session_factory)

    assert drain() == 1  # no exception escapes the worker

    email, job = load(session_factory, email_id)
    assert job.status == JobStatus.QUEUED
    assert job.last_error == "RuntimeError: boom"
    assert email.status == EmailStatus.RECEIVED

    fake_llm.script = [triage_json()]
    make_retries_due(session_factory)
    drain()
    email, job = load(session_factory, email_id)
    assert email.status == EmailStatus.TRIAGED


def test_abandoned_job_goes_back_to_the_queue(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain, settings: Settings
) -> None:
    email_id = add_email(session_factory)
    claim_and_die(session_factory)
    with session_factory() as session:
        released = requeue_stale_jobs(session, lock_timeout_seconds=300)
    assert released == 1

    email, job = load(session_factory, email_id)
    assert job.status == JobStatus.QUEUED
    assert job.locked_by is None
    assert "Lock expired" in (job.last_error or "")

    fake_llm.script = [triage_json()]
    drain()
    email, job = load(session_factory, email_id)
    assert email.status == EmailStatus.TRIAGED
    assert job.attempts == 2  # the abandoned attempt counts


def test_abandoned_job_with_no_attempt_left_fails_the_email(session_factory: Factory) -> None:
    email_id = add_email(session_factory, max_attempts=1)
    claim_and_die(session_factory)
    with session_factory() as session:
        requeue_stale_jobs(session, lock_timeout_seconds=300)

    email, job = load(session_factory, email_id)
    assert job.status == JobStatus.FAILED
    assert email.status == EmailStatus.FAILED


def test_recent_running_job_is_not_released(session_factory: Factory) -> None:
    add_email(session_factory)
    with session_factory() as session:
        claim_next_job(session, worker_id="busy-worker")
        assert requeue_stale_jobs(session, lock_timeout_seconds=300) == 0


def test_job_for_an_email_already_triaged_does_not_call_the_model(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    # A worker triaged the email, then stopped before marking the job done.
    email_id = add_email(session_factory)
    with session_factory() as session:
        session.get_one(Email, email_id).status = EmailStatus.TRIAGED
        session.commit()

    assert drain() == 1

    _, job = load(session_factory, email_id)
    assert job.status == JobStatus.DONE
    assert fake_llm.calls == []


def test_future_retry_is_not_picked_early(session_factory: Factory) -> None:
    add_email(session_factory)
    with session_factory() as session:
        session.execute(update(Job).values(run_after=utcnow() + timedelta(minutes=1)))
        session.commit()
        assert claim_next_job(session, worker_id="w") is None


def test_worker_refuses_to_start_without_a_model(settings: Settings) -> None:
    with pytest.raises(SystemExit):
        main(settings)
