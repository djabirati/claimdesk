"""Deterministic extraction: everything a regex can find reliably, found before any LLM call.

These values serve two purposes: they are given to the model as hints, and they are used
afterwards to check the model's answer. A model can invent an order id; a regex cannot.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal

ORDER_ID_RE = re.compile(r"\bML-?(\d{6})\b", re.IGNORECASE)
# Tracking formats of the (fictional) carriers: 3-letter prefix, 9 digits, "FR".
TRACKING_RE = re.compile(r"\b(RPX|TAL|CNO|HXL)(\d{9})FR\b", re.IGNORECASE)
AMOUNT_RE = re.compile(r"(\d{1,6}(?:[.,]\d{1,2})?)\s?(?:€|eur\b|euros?\b)", re.IGNORECASE)

CARRIER_BY_PREFIX = {
    "RPX": "Rapidex",
    "TAL": "TransAlpes",
    "CNO": "ColiNord",
    "HXL": "Hexa Logistics",
}


@dataclass(frozen=True)
class Extracted:
    order_ids: list[str] = field(default_factory=list)
    tracking_numbers: list[str] = field(default_factory=list)
    amounts_eur: list[Decimal] = field(default_factory=list)

    @property
    def carriers(self) -> list[str]:
        return _unique(CARRIER_BY_PREFIX[t[:3]] for t in self.tracking_numbers)


def _unique[T](items: Iterable[T]) -> list[T]:
    """Remove duplicates, keep the order in which values first appear."""
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract(text: str) -> Extracted:
    # Customers write "ml-104233", "ML104233" or "ML-104233": normalise to ML-104233.
    order_ids = _unique(f"ML-{m.group(1)}" for m in ORDER_ID_RE.finditer(text))
    tracking = _unique(f"{m.group(1).upper()}{m.group(2)}FR" for m in TRACKING_RE.finditer(text))
    amounts = _unique(Decimal(m.group(1).replace(",", ".")) for m in AMOUNT_RE.finditer(text))
    return Extracted(order_ids=order_ids, tracking_numbers=tracking, amounts_eur=amounts)
