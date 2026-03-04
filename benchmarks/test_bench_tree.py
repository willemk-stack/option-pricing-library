from __future__ import annotations

import pytest

from option_pricing.pricers.tree import binom_price_from_ctx
from option_pricing.types import MarketData, OptionType


@pytest.mark.parametrize("n_steps", [200, 1000, 5000, 20000])
def test_bench_binomial_crr_scaling(benchmark, n_steps: int) -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    ctx = market.to_context()

    def _run() -> float:
        return binom_price_from_ctx(
            kind=OptionType.CALL,
            strike=100.0,
            sigma=0.2,
            tau=1.0,
            ctx=ctx,
            n_steps=n_steps,
            american=False,
            method="tree",
        )

    benchmark(_run)
