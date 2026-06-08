from __future__ import annotations

import pytest

from option_pricing.marketdata.config import (
    DIVIDEND_POLICY_IMPLIED_CARRY,
    DIVIDEND_POLICY_MANUAL_STATIC,
    DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD,
    DIVIDEND_POLICY_ZERO_ASSUMPTION,
    MarketDataPolicyConfig,
)
from option_pricing.marketdata.provider_policy import (
    _resolve_provider_snapshot_dividend_policy,
)


def _policy(payload: dict[str, object]) -> MarketDataPolicyConfig:
    return MarketDataPolicyConfig.from_mapping(payload)


def test_manual_static_nonzero_dividend_yield_is_used() -> None:
    policy = _policy(
        {
            "dividends": {
                "static_yields": {
                    "SPY": {
                        "dividend_yield": 0.0125,
                        "source": DIVIDEND_POLICY_MANUAL_STATIC,
                        "note": "fixture yield",
                    }
                }
            }
        }
    )

    resolved = _resolve_provider_snapshot_dividend_policy(
        underlying="spy",
        policy_config=policy,
        dividend_yield=0.0,
        dividend_yield_source=DIVIDEND_POLICY_ZERO_ASSUMPTION,
    )

    assert resolved == {
        "policy": DIVIDEND_POLICY_MANUAL_STATIC,
        "dividend_yield": 0.0125,
        "source": DIVIDEND_POLICY_MANUAL_STATIC,
        "dividend_source": DIVIDEND_POLICY_MANUAL_STATIC,
        "dividend_is_explicit": True,
        "dividend_fallback_used": False,
        "dividend_inference": "not_enabled",
        "dividend_note": "fixture yield",
    }


def test_manual_static_zero_dividend_yield_is_explicit_not_missing() -> None:
    policy = _policy(
        {
            "dividends": {
                "static_yields": {
                    "TSLA": {
                        "dividend_yield": 0.0,
                        "source": DIVIDEND_POLICY_MANUAL_STATIC,
                    }
                }
            }
        }
    )

    resolved = _resolve_provider_snapshot_dividend_policy(
        underlying="TSLA",
        policy_config=policy,
        dividend_yield=0.0,
        dividend_yield_source=DIVIDEND_POLICY_ZERO_ASSUMPTION,
    )

    assert resolved["policy"] == DIVIDEND_POLICY_MANUAL_STATIC
    assert resolved["dividend_yield"] == pytest.approx(0.0)
    assert resolved["dividend_is_explicit"] is True
    assert resolved["dividend_fallback_used"] is False


def test_missing_static_symbol_falls_back_to_zero_assumption() -> None:
    resolved = _resolve_provider_snapshot_dividend_policy(
        underlying="QQQ",
        policy_config=MarketDataPolicyConfig(),
        dividend_yield=0.0,
        dividend_yield_source=DIVIDEND_POLICY_ZERO_ASSUMPTION,
    )

    assert resolved["policy"] == DIVIDEND_POLICY_ZERO_ASSUMPTION
    assert resolved["dividend_yield"] == pytest.approx(0.0)
    assert resolved["dividend_source"] == DIVIDEND_POLICY_ZERO_ASSUMPTION
    assert resolved["dividend_is_explicit"] is False
    assert resolved["dividend_fallback_used"] is True


def test_negative_static_dividend_yield_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        _policy(
            {
                "dividends": {
                    "static_yields": {
                        "SPY": {
                            "dividend_yield": -0.01,
                            "source": DIVIDEND_POLICY_MANUAL_STATIC,
                        }
                    }
                }
            }
        )


@pytest.mark.parametrize(
    ("policy_name", "message"),
    [
        (DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD, "provider_trailing_yield"),
        (DIVIDEND_POLICY_IMPLIED_CARRY, "implied_carry"),
    ],
)
def test_recognized_future_dividend_policies_are_not_implemented(
    policy_name: str,
    message: str,
) -> None:
    policy = _policy({"dividends": {"default_policy": policy_name}})

    with pytest.raises(NotImplementedError, match=message):
        _resolve_provider_snapshot_dividend_policy(
            underlying="SPY",
            policy_config=policy,
            dividend_yield=0.0,
            dividend_yield_source=DIVIDEND_POLICY_ZERO_ASSUMPTION,
        )
