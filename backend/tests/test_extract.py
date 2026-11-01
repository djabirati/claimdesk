from decimal import Decimal

from app.triage.extract import extract


def test_finds_order_id_in_standard_format() -> None:
    assert extract("Ma commande ML-104201 est cassée").order_ids == ["ML-104201"]


def test_normalises_order_id_written_without_dash_or_in_lowercase() -> None:
    assert extract("commande ml104203 et Ml-104204").order_ids == ["ML-104203", "ML-104204"]


def test_removes_duplicate_order_ids_and_keeps_order() -> None:
    text = "ML-104202 ... ML-104201 ... encore ML-104202"
    assert extract(text).order_ids == ["ML-104202", "ML-104201"]


def test_ignores_ids_with_the_wrong_number_of_digits() -> None:
    assert extract("ML-12345 et ML-1234567").order_ids == []


def test_does_not_mistake_an_invoice_number_for_an_order_id() -> None:
    assert extract("facture F-2026-0042, tel 06 12 34 56 78").order_ids == []


def test_finds_tracking_numbers_and_their_carrier() -> None:
    extracted = extract("Suivi : hxl798935572fr puis TAL177777868FR")
    assert extracted.tracking_numbers == ["HXL798935572FR", "TAL177777868FR"]
    assert extracted.carriers == ["Hexa Logistics", "TransAlpes"]


def test_ignores_unknown_tracking_prefixes() -> None:
    assert extract("XYZ123456789FR").tracking_numbers == []


def test_reads_amounts_with_comma_or_dot() -> None:
    assert extract("payé 349 €, puis 27,80 euros et 12.5 EUR").amounts_eur == [
        Decimal("349"),
        Decimal("27.80"),
        Decimal("12.5"),
    ]
