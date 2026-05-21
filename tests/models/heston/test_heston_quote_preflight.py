from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.calibration import (
    HestonQuotePreflight,
    preflight_heston_quotes,
)
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.types import MarketData

MARKET = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)


def _quote_set(
    *,
    strike: list[float],
    expiry: list[float],
    is_call: list[bool],
    mid: list[float],
    bid: list[float] | None = None,
    ask: list[float] | None = None,
) -> HestonQuoteSet:
    return HestonQuoteSet.from_flat_market(
        market=MARKET,
        strike=np.asarray(strike, dtype=np.float64),
        expiry=np.asarray(expiry, dtype=np.float64),
        is_call=np.asarray(is_call, dtype=np.bool_),
        mid=np.asarray(mid, dtype=np.float64),
        bid=None if bid is None else np.asarray(bid, dtype=np.float64),
        ask=None if ask is None else np.asarray(ask, dtype=np.float64),
    )


def test_preflight_valid_mixed_call_put_quotes_pass() -> None:
    quotes = _quote_set(
        strike=[90.0, 110.0],
        expiry=[0.5, 1.0],
        is_call=[True, False],
        mid=[12.0, 11.0],
        bid=[11.5, 10.5],
        ask=[12.5, 11.5],
    )

    result = preflight_heston_quotes(quotes)

    assert isinstance(result, HestonQuotePreflight)
    assert result.quote_count == 2
    assert result.price_bound_violation_count == 0
    assert result.mid_outside_bid_ask_count == 0
    assert result.recommendation == "ok"
    assert result.messages == ("All quotes passed Heston calibration preflight.",)


def test_preflight_flags_call_below_lower_bound() -> None:
    quotes = _quote_set(
        strike=[90.0],
        expiry=[1.0],
        is_call=[True],
        mid=[10.0],
    )

    result = preflight_heston_quotes(quotes)

    assert result.price_bound_violation_count == 1
    assert result.mid_outside_bid_ask_count == 0
    assert result.recommendation == "block"
    assert "vanilla no-arbitrage price bounds" in result.messages[0]


def test_preflight_flags_call_above_upper_bound() -> None:
    quotes = _quote_set(
        strike=[100.0],
        expiry=[1.0],
        is_call=[True],
        mid=[100.0],
    )

    result = preflight_heston_quotes(quotes)

    assert result.price_bound_violation_count == 1
    assert result.recommendation == "block"


def test_preflight_flags_put_below_lower_bound() -> None:
    quotes = _quote_set(
        strike=[120.0],
        expiry=[1.0],
        is_call=[False],
        mid=[18.0],
    )

    result = preflight_heston_quotes(quotes)

    assert result.price_bound_violation_count == 1
    assert result.recommendation == "block"


def test_preflight_flags_put_above_upper_bound() -> None:
    quotes = _quote_set(
        strike=[120.0],
        expiry=[1.0],
        is_call=[False],
        mid=[118.0],
    )

    result = preflight_heston_quotes(quotes)

    assert result.price_bound_violation_count == 1
    assert result.recommendation == "block"


def test_preflight_flags_mid_outside_bid_ask() -> None:
    quotes = _quote_set(
        strike=[100.0],
        expiry=[1.0],
        is_call=[True],
        mid=[4.0],
        bid=[5.0],
        ask=[6.0],
    )

    result = preflight_heston_quotes(quotes)

    assert result.price_bound_violation_count == 0
    assert result.mid_outside_bid_ask_count == 1
    assert result.recommendation == "block"
    assert "mid outside bid/ask" in result.messages[-1]


def test_preflight_raise_on_block_uses_helpful_message() -> None:
    quotes = _quote_set(
        strike=[100.0],
        expiry=[1.0],
        is_call=[True],
        mid=[100.0],
    )

    with pytest.raises(ValueError, match="Heston quote preflight blocked calibration"):
        preflight_heston_quotes(quotes, raise_on_block=True)


def test_preflight_tolerances_allow_tiny_floating_point_noise() -> None:
    base_price_quotes = _quote_set(
        strike=[90.0],
        expiry=[1.0],
        is_call=[True],
        mid=[11.0],
    )
    lower_bound = float(
        base_price_quotes.discount[0]
        * max(base_price_quotes.forward[0] - base_price_quotes.strike[0], 0.0)
    )
    price_bound_quotes = _quote_set(
        strike=[90.0],
        expiry=[1.0],
        is_call=[True],
        mid=[lower_bound - 5.0e-11],
    )
    bid_ask_quotes = _quote_set(
        strike=[100.0],
        expiry=[1.0],
        is_call=[True],
        mid=[6.0 + 5.0e-13],
        bid=[5.0],
        ask=[6.0],
    )

    price_result = preflight_heston_quotes(price_bound_quotes, price_bound_tol=1.0e-10)
    bid_ask_result = preflight_heston_quotes(bid_ask_quotes, bid_ask_tol=1.0e-12)

    assert price_result.recommendation == "ok"
    assert bid_ask_result.recommendation == "ok"
