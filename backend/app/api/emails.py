from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db import get_session
from app.enums import EmailStatus, JobKind
from app.jobs import enqueue
from app.models import Email
from app.schemas import EmailDetail, EmailIn, EmailOut

router = APIRouter(prefix="/emails", tags=["emails"])

SessionDep = Annotated[Session, Depends(get_session)]


@router.post("", response_model=EmailDetail, status_code=status.HTTP_202_ACCEPTED)
def receive_email(payload: EmailIn, session: SessionDep) -> Email:
    """Store an incoming email and queue it for triage.

    The answer comes back at once with status "received". A worker runs the triage in
    the background; GET /emails/{id} shows the result once it is done. A slow or failing
    model call never blocks the caller.
    """
    email = Email(
        sender=payload.sender,
        subject=payload.subject,
        body=payload.body,
        status=EmailStatus.RECEIVED,
    )
    session.add(email)
    enqueue(
        session,
        email=email,
        kind=JobKind.TRIAGE,
        max_attempts=get_settings().job_max_attempts,
    )
    session.commit()  # the email and its job are saved together, or not at all
    return _load_email(session, email.id)


@router.get("", response_model=list[EmailOut])
def list_emails(
    session: SessionDep,
    status_filter: Annotated[EmailStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[Email]:
    query = select(Email).order_by(Email.id.desc()).limit(limit)
    if status_filter is not None:
        query = query.where(Email.status == status_filter)
    return list(session.scalars(query))


@router.get("/{email_id}", response_model=EmailDetail)
def get_email(email_id: int, session: SessionDep) -> Email:
    return _load_email(session, email_id)


def _load_email(session: Session, email_id: int) -> Email:
    """Load an email with its runs and jobs, so the response never triggers a lazy load."""
    email = session.scalar(
        select(Email)
        .where(Email.id == email_id)
        .options(selectinload(Email.runs), selectinload(Email.jobs))
        .execution_options(populate_existing=True)
    )
    if email is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Email not found")
    return email
