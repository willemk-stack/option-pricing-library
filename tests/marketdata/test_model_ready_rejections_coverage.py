from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import option_pricing.marketdata.model_ready as model_ready
from option_pricing.types import MarketData


def test_heston_rejection_reasons_cover_validation_matrix() -> None:
    frame = pd.DataFrame(
        [
            {
                "mid": pd.NA,
                "bid": pd.NA,
                "ask": pd.NA,
                "strike": pd.NA,
                "expiry": pd.NaT,
                "expiry_years": pd.NA,
                "iv": pd.NA,
                "vega": pd.NA,
                "right": pd.NA,
                "option_type": pd.NA,
            },
            {
                "mid": 1.0,
                "bid": 2.0,
                "ask": 1.0,
                "strike": -1.0,
                "expiry": "2026-01-01",
                "expiry_years": 0.0,
                "iv": np.inf,
                "vega": -1.0,
                "right": "call",
                "option_type": "put",
            },
            {
                "mid": 0.0,
                "bid": 1.0,
                "ask": 1.0,
                "strike": 1000.0,
                "expiry": "2026-01-02",
                "expiry_years": 1.0 / 365.0,
                "iv": 0.0,
                "vega": np.nan,
                "right": "put",
                "option_type": "put",
            },
            {
                "mid": 500.0,
                "bid": 499.0,
                "ask": 501.0,
                "strike": 100.0,
                "expiry": "2026-06-01",
                "expiry_years": 30.0 / 365.0,
                "iv": 0.2,
                "vega": np.inf,
                "right": "call",
                "option_type": "call",
            },
        ],
    )

    reasons = model_ready._heston_rejection_reasons(
        frame,
        market_data=MarketData(spot=100.0, rate=0.0, dividend_yield=0.0),
        spot=100.0,
        objective_type="bid_ask_normalized",
        min_expiry_days=7.0,
        max_expiry_days=20.0,
        min_moneyness=0.5,
        max_moneyness=2.0,
        require_iv_for_seed=True,
        require_vega_for_objective=True,
    )
    flat = {reason for row in reasons for reason in row}

    expected = {
        "missing_mid",
        "missing_bid",
        "missing_ask",
        "crossed_market",
        "nonpositive_spread",
        "missing_expiry",
        "nonpositive_time_to_expiry",
        "short_expiry",
        "long_expiry",
        "missing_strike",
        "nonpositive_strike",
        "missing_option_type",
        "invalid_option_type",
        "missing_iv_for_seed",
        "nonfinite_iv",
        "nonpositive_iv",
        "missing_vega_for_objective",
        "nonfinite_vega",
        "nonpositive_vega",
        "nonpositive_mid",
        "price_bound_violation",
        "extreme_moneyness",
    }
    assert expected <= flat


def test_compute_expiry_years_handles_missing_bad_and_timezone_values() -> None:
    assert np.isnan(model_ready._compute_expiry_years(pd.NaT, "2026-01-01"))
    assert np.isnan(model_ready._compute_expiry_years("not-a-date", "2026-01-01"))

    out = model_ready._compute_expiry_years(
        "2026-01-02",
        "2026-01-01T12:00:00+02:00",
    )

    assert out == pytest.approx((14.0 / 24.0) / 365.0)


def test_model_ready_private_validators_and_ordering() -> None:
    assert model_ready._numeric_value("1.25").usable
    assert model_ready._numeric_value("bad").finite is False
    assert model_ready._text_value(" Call ") == "call"
    assert model_ready._text_value(pd.NA) is None
    assert model_ready._dedupe_strings(["a", "b", "a"]) == ("a", "b")
    assert model_ready._rejection_counts(
        [("zzz", "missing_mid"), ("missing_mid", "short_expiry")]
    ) == {"missing_mid": 2, "short_expiry": 1, "zzz": 1}

    with pytest.raises(ValueError, match="requires finite positive vega"):
        model_ready._resolve_require_vega_for_objective("vega_scaled_price", False)

    with pytest.raises(TypeError, match="must be a bool"):
        model_ready._validate_bool("flag", 1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite and nonnegative"):
        model_ready._validate_nonnegative_float("min_expiry_days", -1.0)

    with pytest.raises(ValueError, match="finite and positive"):
        model_ready._validate_optional_positive_float("min_moneyness", 0.0)
