"""The human side: a person approves or rejects what the agent proposed.

This is the only module that executes an action, and the agent never imports it (a test
checks this). Execution is simulated: nothing is paid, shipped or sent; a reference is
recorded as a real system would return one.

Approval is a compare-and-set: UPDATE ... WHERE status = 'proposed'. If two people approve
the same refund at the same moment, Postgres lets one UPDATE through and the other finds
no row left to change, so the refund is executed once.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.enums import ActionKind, ActionStatus
from app.jobs import utcnow
from app.models import ProposedAction


class ActionNotPending(Exception):
    """The action was already approved or rejected."""


def _decide(
    session: Session,
    action_id: int,
    *,
    status: ActionStatus,
    reviewer: str,
    note: str | None,
    now: datetime,
) -> None:
    changed = session.execute(
        update(ProposedAction)
        .where(ProposedAction.id == action_id, ProposedAction.status == ActionStatus.PROPOSED)
        .values(status=status, decided_by=reviewer, decided_at=now, decision_note=note)
    ).rowcount  # type: ignore[attr-defined]
    session.commit()
    if changed == 0:
        raise ActionNotPending(f"Action {action_id} is not waiting for a decision.")


def simulate_execution(action: ProposedAction) -> dict[str, Any]:
    """What the payment, warehouse, carrier or email system would answer."""
    kind = ActionKind(action.kind)
    if kind is ActionKind.REFUND:
        return {
            "refund_reference": f"RF-{action.id:06d}",
            "amount_eur": action.payload["amount_eur"],
        }
    if kind is ActionKind.REPLACEMENT:
        return {"replacement_order": f"ML-R{action.id:05d}"}
    if kind is ActionKind.CARRIER_CLAIM:
        return {"claim_reference": f"CLM-{action.id:06d}", "carrier": action.payload["carrier"]}
    return {"sent_to": action.email.sender, "simulated": True}


def approve_action(
    session: Session, action_id: int, *, reviewer: str, note: str | None = None
) -> ProposedAction:
    """Approve, then execute. Raises ActionNotPending if someone already decided."""
    now = utcnow()
    _decide(session, action_id, status=ActionStatus.APPROVED, reviewer=reviewer, note=note, now=now)
    action = session.get_one(ProposedAction, action_id, populate_existing=True)
    # With real systems this call happens outside any transaction and carries the action
    # id as an idempotency key, so a retry after a crash cannot pay twice.
    action.result = simulate_execution(action)
    action.status = ActionStatus.EXECUTED
    action.executed_at = utcnow()
    session.commit()
    return action


def reject_action(
    session: Session, action_id: int, *, reviewer: str, note: str | None = None
) -> ProposedAction:
    _decide(
        session, action_id, status=ActionStatus.REJECTED, reviewer=reviewer, note=note, now=utcnow()
    )
    return session.get_one(ProposedAction, action_id, populate_existing=True)
