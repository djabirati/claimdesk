"""The agent's tools. Two kinds only, and no tool can change anything outside this run:

- read: look up the sender's orders and the carrier tracking;
- propose: record an action (refund, replacement, carrier claim, reply) for a person to
  approve. Proposals are kept in memory and saved at the end of the run; executing them
  is done by app/actions.py, which this module never imports.

Least privilege: every tool only sees the orders of the email's sender. An email that
asks about someone else's order (or an injected instruction that tries to) gets
"not found", so the agent cannot read or act on another customer's data.

Every argument is validated in code, and an invalid call returns an error to the model
instead of raising: the model can read the error and correct its call.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from email.utils import parseaddr
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, ValidationError, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.carrier import as_utc, track
from app.enums import ActionKind
from app.llm.base import ToolCall, ToolSpec
from app.models import Email, Order
from app.schemas import ORDER_ID_PATTERN

Effect = Literal["read", "propose"]


@dataclass
class Proposal:
    kind: ActionKind
    order_id: str | None
    payload: dict[str, Any]
    rationale: str | None


@dataclass
class ToolContext:
    """What the tools of one run can see: one email, its sender's orders, the proposals."""

    session: Session
    email: Email
    now: datetime
    proposals: list[Proposal] = field(default_factory=list)

    @property
    def customer(self) -> str:
        return parseaddr(self.email.sender)[1].lower()

    def customer_order(self, order_id: str) -> Order | None:
        """The order, only if it belongs to the sender of the email."""
        match = ORDER_ID_PATTERN.match(order_id.strip())
        if not match:
            return None
        return self.session.scalar(
            select(Order).where(
                Order.id == f"ML-{match.group(1)}",
                func.lower(Order.customer_email) == self.customer,
            )
        )

    def already_proposed(self, kind: ActionKind) -> bool:
        return any(p.kind is kind for p in self.proposals)


# ---------- argument models (also the JSON schemas sent to the model) ----------
# Kept flat with plain types: constraints are checked in validators, not in the schema,
# so the schema stays within what every provider supports.


class NoArgs(BaseModel):
    pass


class OrderArgs(BaseModel):
    order_id: str = Field(description="Order id, format ML-XXXXXX.")


class TrackingArgs(BaseModel):
    tracking_number: str = Field(description="Tracking number of one of the customer's orders.")


def _reason(value: str) -> str:
    value = value.strip()
    if not 5 <= len(value) <= 500:
        raise ValueError("reason must be between 5 and 500 characters")
    return value


Reason = Annotated[str, AfterValidator(_reason)]


class RefundArgs(BaseModel):
    order_id: str = Field(description="Order id, format ML-XXXXXX.")
    amount_eur: float = Field(description="Amount to refund in euros, at most the order total.")
    reason: Reason = Field(description="Why this refund, in one sentence, for the reviewer.")


class OrderReasonArgs(BaseModel):
    order_id: str = Field(description="Order id, format ML-XXXXXX.")
    reason: Reason = Field(description="Why this action, in one sentence, for the reviewer.")


class ReplyArgs(BaseModel):
    message: str = Field(
        description="The full reply to the customer, in the customer's language, signed "
        "'Le service client Maison Lumen'."
    )

    @field_validator("message")
    @classmethod
    def message_length(cls, value: str) -> str:
        value = value.strip()
        if not 20 <= len(value) <= 2000:
            raise ValueError("message must be between 20 and 2000 characters")
        return value


# ---------- tools ----------

NOT_FOUND = {"error": "No order with this id for this customer."}
PROPOSED = {"status": "proposed", "note": "Recorded for human approval. Nothing was executed."}


def _order_summary(order: Order, now: datetime) -> dict[str, Any]:
    delivered = as_utc(order.delivered_at) if order.delivered_at else None
    return {
        "order_id": order.id,
        "customer_name": order.customer_name,
        "total_eur": str(order.total_amount),
        "status": order.status,
        "carrier": order.carrier,
        "tracking_number": order.tracking_number,
        "shipped_at": as_utc(order.shipped_at).date().isoformat(),
        "delivered_at": delivered.date().isoformat() if delivered else None,
        "days_since_delivery": (now - delivered).days if delivered else None,
    }


def get_order(ctx: ToolContext, args: OrderArgs) -> dict[str, Any]:
    order = ctx.customer_order(args.order_id)
    return _order_summary(order, ctx.now) if order else NOT_FOUND


def list_customer_orders(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    orders = ctx.session.scalars(
        select(Order)
        .where(func.lower(Order.customer_email) == ctx.customer)
        .order_by(Order.shipped_at.desc())
    )
    return {"orders": [_order_summary(order, ctx.now) for order in orders]}


def track_parcel(ctx: ToolContext, args: TrackingArgs) -> dict[str, Any]:
    order = ctx.session.scalar(
        select(Order).where(
            Order.tracking_number == args.tracking_number.strip().upper(),
            func.lower(Order.customer_email) == ctx.customer,
        )
    )
    if order is None:
        return {"error": "No parcel with this tracking number for this customer."}
    return track(order, now=ctx.now)


def propose_refund(ctx: ToolContext, args: RefundArgs) -> dict[str, Any]:
    order = ctx.customer_order(args.order_id)
    if order is None:
        return NOT_FOUND
    if ctx.already_proposed(ActionKind.REFUND):
        return {"error": "A refund is already proposed for this email."}
    try:
        amount = Decimal(str(args.amount_eur)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return {"error": "amount_eur must be a number."}
    if amount <= 0:
        return {"error": "amount_eur must be greater than 0."}
    if amount > order.total_amount:
        return {"error": f"amount_eur cannot exceed the order total ({order.total_amount} EUR)."}
    ctx.proposals.append(
        Proposal(ActionKind.REFUND, order.id, {"amount_eur": str(amount)}, args.reason)
    )
    return PROPOSED


def propose_replacement(ctx: ToolContext, args: OrderReasonArgs) -> dict[str, Any]:
    order = ctx.customer_order(args.order_id)
    if order is None:
        return NOT_FOUND
    if ctx.already_proposed(ActionKind.REPLACEMENT):
        return {"error": "A replacement is already proposed for this email."}
    ctx.proposals.append(Proposal(ActionKind.REPLACEMENT, order.id, {}, args.reason))
    return PROPOSED


def propose_carrier_claim(ctx: ToolContext, args: OrderReasonArgs) -> dict[str, Any]:
    order = ctx.customer_order(args.order_id)
    if order is None:
        return NOT_FOUND
    if ctx.already_proposed(ActionKind.CARRIER_CLAIM):
        return {"error": "A carrier claim is already proposed for this email."}
    payload = {"carrier": order.carrier, "tracking_number": order.tracking_number}
    ctx.proposals.append(Proposal(ActionKind.CARRIER_CLAIM, order.id, payload, args.reason))
    return PROPOSED


def propose_reply(ctx: ToolContext, args: ReplyArgs) -> dict[str, Any]:
    if ctx.already_proposed(ActionKind.REPLY):
        return {"error": "A reply is already proposed for this email."}
    ctx.proposals.append(Proposal(ActionKind.REPLY, None, {"message": args.message}, None))
    return PROPOSED


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    effect: Effect
    args: type[BaseModel]
    run: Callable[[ToolContext, Any], dict[str, Any]]

    @property
    def spec(self) -> ToolSpec:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("properties", {})
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


TOOLS: list[Tool] = [
    Tool(
        "get_order",
        "Details of one of the customer's orders: total, status, carrier, tracking, dates.",
        "read",
        OrderArgs,
        get_order,
    ),
    Tool(
        "list_customer_orders",
        "All orders of the customer who sent the email, newest first. Use it when the email "
        "gives no order id.",
        "read",
        NoArgs,
        list_customer_orders,
    ),
    Tool(
        "track_parcel",
        "Carrier tracking of one of the customer's parcels: events and days since the last one.",
        "read",
        TrackingArgs,
        track_parcel,
    ),
    Tool(
        "propose_refund",
        "Propose a refund, at most the order total. A person approves it before anything happens.",
        "propose",
        RefundArgs,
        propose_refund,
    ),
    Tool(
        "propose_replacement",
        "Propose to send the product again. A person approves it before anything happens.",
        "propose",
        OrderReasonArgs,
        propose_replacement,
    ),
    Tool(
        "propose_carrier_claim",
        "Propose to open a claim with the carrier (damaged, lost or stuck parcel). A person "
        "approves it before anything happens.",
        "propose",
        OrderReasonArgs,
        propose_carrier_claim,
    ),
    Tool(
        "propose_reply",
        "Propose the reply to send to the customer. A person approves it before it is sent.",
        "propose",
        ReplyArgs,
        propose_reply,
    ),
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
TOOL_SPECS = [tool.spec for tool in TOOLS]

_SPACES = re.compile(r"\s+")


def _short_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors()
    )


def call_tool(ctx: ToolContext, call: ToolCall) -> tuple[dict[str, Any], int]:
    """Run one tool call. Never raises: errors go back to the model as {"error": ...}.

    Returns the result and the duration in milliseconds.
    """
    start = time.perf_counter()
    tool = TOOLS_BY_NAME.get(call.name)
    if tool is None:
        result: dict[str, Any] = {"error": f"Unknown tool: {_SPACES.sub(' ', call.name)[:60]}"}
    else:
        try:
            args = tool.args.model_validate(call.arguments)
        except ValidationError as exc:
            result = {"error": f"Invalid arguments: {_short_errors(exc)}"}
        else:
            result = tool.run(ctx, args)
    return result, round((time.perf_counter() - start) * 1000)
