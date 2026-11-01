"""Load the fictional orders into the database. Safe to run several times.

Usage (from backend/):  python -m scripts.seed
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from app.db import SessionLocal
from app.models import Order

SEED_FILE = Path(__file__).resolve().parents[2] / "data" / "seed" / "orders.json"


def main() -> None:
    rows = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    with SessionLocal() as session:
        for row in rows:
            order = Order(
                id=row["id"],
                customer_name=row["customer_name"],
                customer_email=row["customer_email"],
                total_amount=Decimal(row["total_amount"]),
                carrier=row["carrier"],
                tracking_number=row["tracking_number"],
                status=row["status"],
                shipped_at=datetime.fromisoformat(row["shipped_at"]),
                delivered_at=(
                    datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None
                ),
            )
            session.merge(order)  # insert, or update if the id already exists
        session.commit()
    print(f"Seeded {len(rows)} orders.")


if __name__ == "__main__":
    main()
