from decimal import Decimal

from app.config import Settings


def estimate_cost(input_tokens: int, output_tokens: int, settings: Settings) -> Decimal:
    """Estimated cost in USD, from the per-million-token prices in the settings."""
    cost = (
        input_tokens * settings.llm_input_price_per_mtok
        + output_tokens * settings.llm_output_price_per_mtok
    ) / 1_000_000
    return Decimal(str(round(cost, 6)))
