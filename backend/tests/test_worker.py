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
from app.llm.fake import FakeLLMClient, model_turn
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
    assert drain() == 2  # the triage succeeds, then the agent job of the claim runs

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


def test_rejected_request_fails_at_once_without_retry(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    # What happened on the first real run: the model was closed to new users (404).
    fake_llm.script = [LLMError("404 NOT_FOUND: model not available", retryable=False)]
    email_id = add_email(session_factory)

    drain()

    email, job = load(session_factory, email_id)
    assert job.status == JobStatus.FAILED
    assert job.attempts == 1
    assert "404" in (job.last_error or "")
    assert email.status == EmailStatus.FAILED
    assert [run.status for run in email.runs] == ["rejected"]
    assert len(fake_llm.calls) == 1


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


# ---------- agent jobs ----------


def jobs_of(session_factory: Factory, email_id: int) -> list[tuple[str, str, int]]:
    with session_factory() as session:
        email = session.get_one(Email, email_id)
        return [(job.kind, job.status, job.attempts) for job in email.jobs]


def test_a_claim_gets_an_agent_job_after_its_triage(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json()]
    email_id = add_email(session_factory)

    assert drain() == 2

    assert jobs_of(session_factory, email_id) == [("triage", "done", 1), ("agent", "done", 1)]


def test_an_email_that_is_not_a_claim_gets_no_agent_job(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [
        triage_json(category="not_a_claim", requested_action="none", order_id=None, urgency="low")
    ]
    email_id = add_email(session_factory)

    assert drain() == 1

    assert jobs_of(session_factory, email_id) == [("triage", "done", 1)]


def test_an_agent_provider_error_is_retried_without_failing_the_email(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json()]
    fake_llm.turns = [LLMError("503 overloaded"), model_turn(text="Fini.")]
    email_id = add_email(session_factory)

    drain()
    assert jobs_of(session_factory, email_id) == [("triage", "done", 1), ("agent", "queued", 1)]
    make_retries_due(session_factory)
    drain()

    assert jobs_of(session_factory, email_id) == [("triage", "done", 1), ("agent", "done", 2)]
    with session_factory() as session:
        email = session.get_one(Email, email_id)
        assert email.status == EmailStatus.TRIAGED
        assert [run.status for run in email.agent_runs] == ["error", "completed"]


def test_a_rejected_agent_job_fails_but_keeps_the_triage(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json()]
    fake_llm.turns = [LLMError("400 bad request", retryable=False)]
    email_id = add_email(session_factory)

    drain()

    assert jobs_of(session_factory, email_id) == [("triage", "done", 1), ("agent", "failed", 1)]
    with session_factory() as session:
        assert session.get_one(Email, email_id).status == EmailStatus.TRIAGED


def test_an_agent_job_that_comes_back_after_its_run_is_not_run_twice(
    session_factory: Factory, fake_llm: FakeLLMClient, drain: Drain
) -> None:
    fake_llm.script = [triage_json()]
    email_id = add_email(session_factory)
    drain()
    # A worker saved the agent run, then died before marking the job done.
    with session_factory() as session:
        session.execute(update(Job).where(Job.kind == "agent").values(status="queued"))
        session.commit()
    calls_before = len(fake_llm.calls)

    assert drain() == 1

    assert len(fake_llm.calls) == calls_before  # no new model call
    with session_factory() as session:
        assert len(session.get_one(Email, email_id).agent_runs) == 1
