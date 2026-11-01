"""A job queue stored in Postgres, in the same database as the emails.

How a worker gets work: `claim_next_job` selects the oldest due job with
SELECT ... FOR UPDATE SKIP LOCKED. FOR UPDATE locks the row; SKIP LOCKED makes another
worker skip a row that is being claimed instead of waiting for it. Two workers never get
the same job, and none of them blocks the others.

The claim is committed at once (status = running), so no transaction stays open during
a slow model call. The price: if a worker dies while running a job, nothing releases it
automatically. `requeue_stale_jobs` puts such jobs back in the queue once their lock is
older than the lock timeout. A job can therefore run twice, so the work it does must be
safe to repeat (see app/worker.py).
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.enums import EmailStatus, JobKind, JobStatus
from app.models import Email, Job


def utcnow() -> datetime:
    return datetime.now(UTC)


def enqueue(
    session: Session, *, email: Email, kind: JobKind, max_attempts: int, now: datetime | None = None
) -> Job:
    """Add a job for an email. Does not commit: the caller commits the email and its job
    together, so an email is never stored without the job that will process it."""
    job = Job(
        email=email,
        kind=kind,
        status=JobStatus.QUEUED,
        attempts=0,
        max_attempts=max_attempts,
        run_after=now or utcnow(),
    )
    session.add(job)
    return job


def claim_next_job(session: Session, *, worker_id: str, now: datetime | None = None) -> Job | None:
    """Take the oldest job that is due, mark it running and commit. None if there is none."""
    now = now or utcnow()
    job = session.scalar(
        select(Job)
        .where(Job.status == JobStatus.QUEUED, Job.run_after <= now)
        .order_by(Job.run_after, Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if job is None:
        session.rollback()  # close the transaction opened by the SELECT
        return None
    job.status = JobStatus.RUNNING
    job.attempts += 1
    job.locked_by = worker_id
    job.locked_at = now
    session.commit()
    return job


def mark_done(session: Session, job: Job, *, now: datetime | None = None) -> None:
    job.status = JobStatus.DONE
    job.finished_at = now or utcnow()
    session.commit()


def mark_failed(
    session: Session, job: Job, error: str | None, *, now: datetime | None = None
) -> None:
    """Give up on a job. Its email is marked failed so that a person sees it."""
    job.status = JobStatus.FAILED
    job.last_error = error
    job.finished_at = now or utcnow()
    job.email.status = EmailStatus.FAILED
    session.commit()


def retry_delay(attempts: int, base_seconds: float) -> timedelta:
    """Wait longer after each failure: base, then x4 each time (15 s, 1 min, 4 min...).

    A provider that refuses calls (quota, overload) needs time to recover; retrying at
    once would only add load and fail again.
    """
    return timedelta(seconds=base_seconds * 4 ** (attempts - 1))


def schedule_retry(
    session: Session,
    job: Job,
    error: str | None,
    *,
    base_seconds: float,
    now: datetime | None = None,
) -> bool:
    """Put the job back in the queue for later. Returns False (and fails the job) when no
    attempt is left."""
    now = now or utcnow()
    if job.attempts >= job.max_attempts:
        mark_failed(session, job, error, now=now)
        return False
    job.status = JobStatus.QUEUED
    job.run_after = now + retry_delay(job.attempts, base_seconds)
    job.last_error = error
    job.locked_by = None
    job.locked_at = None
    session.commit()
    return True


def requeue_stale_jobs(
    session: Session, *, lock_timeout_seconds: float, now: datetime | None = None
) -> int:
    """Release jobs left running by a worker that crashed or was killed.

    Their attempt still counts, so a job that keeps crashing workers ends up failed
    instead of looping forever. Returns the number of jobs released.
    """
    now = now or utcnow()
    cutoff = now - timedelta(seconds=lock_timeout_seconds)
    stale = list(
        session.scalars(
            select(Job)
            .where(Job.status == JobStatus.RUNNING, Job.locked_at < cutoff)
            .with_for_update(skip_locked=True)
        )
    )
    error = "Lock expired: the worker stopped while running this job."
    for job in stale:
        if job.attempts >= job.max_attempts:
            job.status = JobStatus.FAILED
            job.finished_at = now
            job.email.status = EmailStatus.FAILED
        else:
            job.status = JobStatus.QUEUED
            job.run_after = now
            job.locked_by = None
            job.locked_at = None
        job.last_error = error
    session.commit()
    return len(stale)
