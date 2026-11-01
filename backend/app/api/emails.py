from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_llm
from app.config import get_settings
from app.db import get_session
from app.enums import EmailStatus
from app.llm.base import LLMClient
from app.models import Email
from app.schemas import EmailDetail, EmailIn, EmailOut
from app.triage.pipeline import triage_email

router = APIRouter(prefix="/emails", tags=["emails"])

SessionDep = Annotated[Session, Depends(get_session)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]


@router.post("", response_model=EmailDetail, status_code=status.HTTP_201_CREATED)
def receive_email(payload: EmailIn, session: SessionDep, llm: LLMDep) -> Email:
    """Store an incoming email and triage it.

    Triage runs inside the request for now. Step 2 moves it to a queue and a worker,
    so a slow or failing model call no longer blocks the caller.
    """
    email = Email(sender=payload.sender, subject=payload.subject, body=payload.body)
    session.add(email)
    session.commit()
    triage_email(session, email, llm, get_settings())
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
    """Load an email with its runs, so the response never triggers a lazy load."""
    email = session.scalar(
        select(Email)
        .where(Email.id == email_id)
        .options(selectinload(Email.runs))
        .execution_options(populate_existing=True)
    )
    if email is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Email not found")
    return email
