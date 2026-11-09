"""Simulated carrier tracking API.

The four carriers are fictional, so their tracking API is simulated from the orders table:
the events follow the order's dates and status. The agent reaches it through a read-only
tool, the same way it would reach a real carrier API.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from app.models import Order


def as_utc(value: datetime) -> datetime:
    """SQLite (unit tests) returns naive datetimes; Postgres returns aware ones."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def track(order: Order, *, now: datetime) -> dict[str, Any]:
    """Tracking events of an order's parcel, as a carrier API would return them."""
    shipped = as_utc(order.shipped_at)
    events = [
        (shipped, "picked_up", f"Colis pris en charge par {order.carrier}"),
        (shipped + timedelta(days=1), "in_transit", "En transit : arrivé sur la plateforme de tri"),
    ]
    if order.status == "delivered" and order.delivered_at is not None:
        events.append((as_utc(order.delivered_at), "delivered", "Colis livré"))
    elif order.status == "returned":
        returned_at = as_utc(order.delivered_at) if order.delivered_at else shipped + timedelta(3)
        events.append((returned_at, "returned", "Colis retourné à l'expéditeur"))

    past = [event for event in events if event[0] <= now] or events[:1]
    last_at = past[-1][0]
    return {
        "tracking_number": order.tracking_number,
        "carrier": order.carrier,
        "status": past[-1][1],
        "events": [
            {"at": at.isoformat(), "status": status, "description": description}
            for at, status, description in past
        ],
        # Computed here: date arithmetic is not something to leave to the model.
        "days_since_last_event": (now - last_at).days,
    }
