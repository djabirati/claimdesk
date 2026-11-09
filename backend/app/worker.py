"""Background worker: takes jobs from the Postgres queue and runs them.

Two kinds of jobs: `triage` for every new email, then `agent` for an email triaged as a
claim. The agent job is queued in the same transaction that marks the triage done.

Usage (from backend/, with the database migrated and GEMINI_API_KEY set):
    python -m app.worker

Several workers can run side by side (docker compose up --scale worker=3): the queue
guarantees that each job goes to one worker at a time. On SIGTERM or Ctrl-C the worker
finishes its current job, then stops.
"""

import logging
import os
import signal
import socket
import threading

from sqlalchemy.orm import Session, sessionmaker

from app.agent.loop import run_agent
from app.config import Settings, get_settings
from app.db import SessionLocal
from app.enums import CLAIM_CATEGORIES, AgentRunStatus, EmailStatus, JobKind, RunStatus
from app.jobs import (
    claim_next_job,
    enqueue,
    mark_done,
    mark_failed,
    requeue_stale_jobs,
    schedule_retry,
)
from app.llm.base import LLMClient, LLMNotConfiguredError
from app.llm.factory import build_llm_client
from app.models import Job
from app.triage.pipeline import triage_email

log = logging.getLogger("claimdesk.worker")


def process_job(session: Session, job: Job, llm: LLMClient, settings: Settings) -> None:
    """Run one claimed job and record what happened to it."""
    if job.kind == JobKind.AGENT:
        process_agent_job(session, job, llm, settings)
    else:
        process_triage_job(session, job, llm, settings)


def process_triage_job(session: Session, job: Job, llm: LLMClient, settings: Settings) -> None:
    email = job.email
    if email.status != EmailStatus.RECEIVED:
        # Already triaged: a worker probably stopped after the triage and before marking
        # the job done, and the job came back. Do not pay for a second model call.
        mark_done(session, job)
        log.info("job %s: email %s already %s, nothing to do", job.id, email.id, email.status)
        return

    run = triage_email(session, email, llm, settings)

    # run.status is read back from the database as a plain string: compare with ==, not `is`.
    if run.status == RunStatus.SUCCESS:
        category = (email.triage or {}).get("category")
        if category in CLAIM_CATEGORIES:
            # Saved by the same commit as "triage done": a claim never misses its agent run.
            enqueue(session, email=email, kind=JobKind.AGENT, max_attempts=job.max_attempts)
        mark_done(session, job)
        log.info("job %s: email %s -> %s (%d ms)", job.id, email.id, email.status, run.latency_ms)
    elif run.status == RunStatus.ERROR:
        # The provider did not answer (network, quota, server error): worth trying again later.
        if schedule_retry(session, job, run.error, base_seconds=settings.job_retry_base_seconds):
            log.warning("job %s: provider error, retry at %s: %s", job.id, job.run_after, run.error)
        else:
            log.error("job %s: provider error, no attempt left: %s", job.id, run.error)
    elif run.status == RunStatus.REJECTED:
        # The provider refused the request (invalid key, unknown model): every retry would
        # fail the same way. Fail now so the error is visible at once.
        mark_failed(session, job, run.error)
        log.error("job %s: request rejected by the provider: %s", job.id, run.error)
    else:
        # The model answered, but twice with invalid output (classify() already retried once
        # with the validation error). The same email would most likely fail the same way.
        mark_failed(session, job, run.error)
        log.error("job %s: invalid model output, email %s failed: %s", job.id, email.id, run.error)


def process_agent_job(session: Session, job: Job, llm: LLMClient, settings: Settings) -> None:
    email = job.email
    finished = (AgentRunStatus.COMPLETED, AgentRunStatus.STEP_LIMIT)
    if any(run.status in finished for run in email.agent_runs):
        # The job came back after a crash, but its run was saved: do not propose twice.
        mark_done(session, job)
        log.info("job %s: email %s already has an agent run, nothing to do", job.id, email.id)
        return

    run = run_agent(session, email, llm, settings)

    if run.status in finished:
        mark_done(session, job)
        log.info(
            "job %s: agent %s on email %s, %d step(s), %d proposal(s)",
            job.id,
            run.status,
            email.id,
            run.steps,
            len(run.actions),
        )
    elif run.status == AgentRunStatus.ERROR:
        if schedule_retry(session, job, run.error, base_seconds=settings.job_retry_base_seconds):
            log.warning("job %s: provider error, retry at %s: %s", job.id, job.run_after, run.error)
        else:
            log.error("job %s: provider error, no attempt left: %s", job.id, run.error)
    else:
        mark_failed(session, job, run.error)
        log.error("job %s: request rejected by the provider: %s", job.id, run.error)


def run_once(
    session_factory: sessionmaker[Session], llm: LLMClient, settings: Settings, worker_id: str
) -> bool:
    """Process at most one job. Returns True if a job was taken from the queue."""
    with session_factory() as session:
        released = requeue_stale_jobs(
            session, lock_timeout_seconds=settings.job_lock_timeout_seconds
        )
        if released:
            log.warning("released %d abandoned job(s)", released)

        job = claim_next_job(session, worker_id=worker_id)
        if job is None:
            return False

        try:
            process_job(session, job, llm, settings)
        except Exception as exc:
            # A bug or a database error. Keep the worker alive and retry the job later;
            # if it keeps failing, it ends up failed instead of blocking the queue.
            session.rollback()
            log.exception("job %s: unexpected error", job.id)
            schedule_retry(
                session,
                job,
                f"{type(exc).__name__}: {exc}",
                base_seconds=settings.job_retry_base_seconds,
            )
        return True


def run_forever(
    session_factory: sessionmaker[Session],
    llm: LLMClient,
    settings: Settings,
    worker_id: str,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            worked = run_once(session_factory, llm, settings, worker_id)
        except Exception:
            # Most likely the database is unreachable. Wait, then try again.
            log.exception("worker loop error")
            worked = False
        if not worked:
            stop.wait(settings.worker_poll_seconds)  # returns at once when stop is set


def main(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = settings or get_settings()
    try:
        llm = build_llm_client(settings)
    except LLMNotConfiguredError as exc:
        # Fail at start-up rather than failing every job one by one.
        log.error("cannot start the worker: %s", exc)
        raise SystemExit(1) from exc

    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    log.info("worker %s started, model %s", worker_id, llm.model)
    run_forever(SessionLocal, llm, settings, worker_id, stop)
    log.info("worker %s stopped", worker_id)


if __name__ == "__main__":
    main()
