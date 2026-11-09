"""The approval queue: what the agent proposed, and a person's decision on each proposal."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.actions import ActionNotPending, approve_action, reject_action
from app.db import get_session
from app.enums import ActionStatus
from app.models import ProposedAction
from app.schemas import ActionOut, DecisionIn

router = APIRouter(prefix="/actions", tags=["actions"])

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("", response_model=list[ActionOut])
def list_actions(
    session: SessionDep,
    status_filter: Annotated[ActionStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ProposedAction]:
    """GET /actions?status=proposed is the approval queue, oldest first."""
    query = select(ProposedAction).order_by(ProposedAction.id).limit(limit)
    if status_filter is not None:
        query = query.where(ProposedAction.status == status_filter)
    return list(session.scalars(query))


@router.get("/{action_id}", response_model=ActionOut)
def get_action(action_id: int, session: SessionDep) -> ProposedAction:
    return _get_or_404(session, action_id)


@router.post("/{action_id}/approve", response_model=ActionOut)
def approve(action_id: int, decision: DecisionIn, session: SessionDep) -> ProposedAction:
    """Approve the action and execute it (simulated). 409 if it was already decided."""
    _get_or_404(session, action_id)
    try:
        return approve_action(session, action_id, reviewer=decision.reviewer, note=decision.note)
    except ActionNotPending as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/{action_id}/reject", response_model=ActionOut)
def reject(action_id: int, decision: DecisionIn, session: SessionDep) -> ProposedAction:
    """Reject the action: it will never run. 409 if it was already decided."""
    _get_or_404(session, action_id)
    try:
        return reject_action(session, action_id, reviewer=decision.reviewer, note=decision.note)
    except ActionNotPending as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _get_or_404(session: Session, action_id: int) -> ProposedAction:
    action = session.get(ProposedAction, action_id)
    if action is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Action not found")
    return action
