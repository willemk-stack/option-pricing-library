from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import option_pricing.marketdata.model_ready as model_ready
from option_pricing.marketdata import prepare_heston_market_fit
from option_pricing.marketdata.bundles import LoadedModelValidationBundle
from option_pricing.marketdata.gold import GoldMarketDataSnapshot
from option_pricing.marketdata.model_ready import (
    HestonReadyStats,
    PreparedHestonMarketFit,
)
from option_pricing.marketdata.schemas import HESTON_QUOTES_COLUMNS, DatasetName
from option_pricing.marketdata.validation import coerce_frame
from option_pricing.models.heston.calibration.preflight import HestonQuotePreflight
from option_pricing.types import MarketData

ASOF = "2026-05-22T15:30:00Z"


def _market_data() -> MarketData:
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)


def _base_records() -> list[dict[str, object]]:
    return [
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260621C00095000",
            "quote_id": "quote-call-95",
            "asof": ASOF,
            "expiry": "2026-06-21",
            "expiry_years": 30.0 / 365.0,
            "strike": 95.0,
            "right": "call",
            "mid": 7.0,
            "bid": 6.9,
            "ask": 7.1,
            "iv": 0.25,
            "vega": 0.08,
            "option_type": "call",
            "label": "SYNTH260621C00095000",
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260721P00105000",
            "quote_id": "quote-put-105",
            "asof": ASOF,
            "expiry": "2026-07-21",
            "expiry_years": 60.0 / 365.0,
            "strike": 105.0,
            "right": "put",
            "mid": 6.5,
            "bid": 6.4,
            "ask": 6.6,
            "iv": 0.28,
            "vega": 0.09,
            "option_type": "put",
            "label": "SYNTH260721P00105000",
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260820C00100000",
            "quote_id": "quote-call-100",
            "asof": ASOF,
            "expiry": "2026-08-20",
            "expiry_years": 90.0 / 365.0,
            "strike": 100.0,
            "right": "call",
            "mid": 4.2,
            "bid": 4.1,
            "ask": 4.3,
            "iv": 0.24,
            "vega": 0.07,
            "option_type": "call",
            "label": "SYNTH260820C00100000",
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
    ]


def _heston_quotes(
    row_overrides: dict[int, dict[str, object]] | None = None,
    *,
    drop_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    records = _base_records()
    for row_index, overrides in (row_overrides or {}).items():
        records[row_index].update(overrides)
    frame = pd.DataFrame(records)
    if drop_columns:
        return frame.drop(columns=list(drop_columns))
    return coerce_frame(frame, DatasetName.HESTON_QUOTES, allow_extra=False)


def _bundle(
    tmp_path: Path,
    heston_quotes: pd.DataFrame,
    *,
    market_data: MarketData | None = None,
) -> LoadedModelValidationBundle:
    market_data = _market_data() if market_data is None else market_data
    return LoadedModelValidationBundle(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={"artifact_schema_version": "model_validation_bundle.v1"},
        warnings={"warnings": []},
        market_snapshot=GoldMarketDataSnapshot(market_data=market_data, metadata={}),
        market_data=market_data,
        cleaned_quotes=pd.DataFrame(),
        rejected_quotes=pd.DataFrame(),
        heston_quotes=heston_quotes,
        surface_inputs=pd.DataFrame(),
        heston_fit_summary=pd.DataFrame(),
    )


def _flatten_reasons(rejected_quotes: pd.DataFrame) -> list[str]:
    return [
        reason for reasons in rejected_quotes["reject_reasons"] for reason in reasons
    ]


def test_public_result_dataclasses_have_expected_fields() -> None:
    assert tuple(field.name for field in fields(HestonReadyStats)) == (
        "input_quote_count",
        "selected_quote_count",
        "rejected_quote_count",
        "rejection_counts",
        "expiry_count",
        "min_expiry_days",
        "max_expiry_days",
        "warnings",
    )
    assert tuple(field.name for field in fields(PreparedHestonMarketFit)) == (
        "model_name",
        "objective_type",
        "market_data",
        "selected_quotes",
        "rejected_quotes",
        "quote_set",
        "preflight",
        "stats",
        "status",
        "warnings",
    )


def test_prepare_heston_market_fit_success_from_loaded_bundle(tmp_path: Path) -> None:
    quotes = _heston_quotes()
    original = quotes.copy(deep=True)

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert isinstance(prepared, PreparedHestonMarketFit)
    assert prepared.status == "ready"
    assert prepared.model_name == "heston"
    assert prepared.objective_type == "price_rmse"
    assert tuple(prepared.selected_quotes.columns) == HESTON_QUOTES_COLUMNS
    assert prepared.rejected_quotes.empty
    assert prepared.quote_set is not None
    assert prepared.quote_set.n_quotes == len(quotes)
    assert prepared.preflight is not None
    assert prepared.preflight.recommendation == "ok"
    assert prepared.stats.input_quote_count == len(quotes)
    assert prepared.stats.selected_quote_count == len(quotes)
    assert prepared.stats.rejected_quote_count == 0
    assert prepared.stats.rejection_counts == {}
    assert prepared.stats.expiry_count == 3
    assert prepared.stats.min_expiry_days == pytest.approx(30.0)
    assert prepared.stats.max_expiry_days == pytest.approx(90.0)
    pd.testing.assert_frame_equal(quotes, original)


def test_default_objective_is_price_rmse(tmp_path: Path) -> None:
    prepared = prepare_heston_market_fit(_bundle(tmp_path, _heston_quotes()))

    assert prepared.objective_type == "price_rmse"


def test_price_rmse_does_not_require_vega(tmp_path: Path) -> None:
    quotes = _heston_quotes({0: {"vega": pd.NA}, 1: {"vega": pd.NA}})

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert prepared.status == "ready"
    assert prepared.rejected_quotes.empty
    assert prepared.quote_set is not None
    assert prepared.quote_set.bs_vega is None


def test_vega_scaled_price_rejects_missing_and_nonpositive_vega(
    tmp_path: Path,
) -> None:
    quotes = _heston_quotes({0: {"vega": pd.NA}, 1: {"vega": 0.0}})

    prepared = prepare_heston_market_fit(
        _bundle(tmp_path, quotes),
        objective_type="vega_scaled_price",
    )

    assert prepared.status == "ready"
    assert len(prepared.selected_quotes) == 1
    assert prepared.stats.rejection_counts["missing_vega_for_objective"] == 1
    assert prepared.stats.rejection_counts["nonpositive_vega"] == 1
    assert (
        "missing_vega_for_objective"
        in prepared.rejected_quotes.loc[
            0,
            "reject_reasons",
        ]
    )
    assert "nonpositive_vega" in prepared.rejected_quotes.loc[1, "reject_reasons"]


def test_missing_iv_is_rejected_when_required_for_seed(tmp_path: Path) -> None:
    quotes = _heston_quotes({0: {"iv": pd.NA}})

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert len(prepared.selected_quotes) == 2
    assert prepared.stats.rejection_counts == {"missing_iv_for_seed": 1}
    assert prepared.rejected_quotes.loc[0, "reject_reasons"] == ("missing_iv_for_seed",)


def test_missing_iv_is_not_rejected_when_seed_iv_not_required(tmp_path: Path) -> None:
    quotes = _heston_quotes({0: {"iv": pd.NA}})

    prepared = prepare_heston_market_fit(
        _bundle(tmp_path, quotes),
        require_iv_for_seed=False,
    )

    assert prepared.status == "ready"
    assert prepared.rejected_quotes.empty
    assert prepared.quote_set is not None
    assert prepared.quote_set.iv_mid is None
    assert any("IV" in warning for warning in prepared.warnings)


def test_short_expiry_is_rejected(tmp_path: Path) -> None:
    quotes = _heston_quotes({0: {"expiry_years": 3.0 / 365.0}})

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert len(prepared.selected_quotes) == 2
    assert prepared.stats.rejection_counts == {"short_expiry": 1}
    assert prepared.rejected_quotes.loc[0, "reject_reasons"] == ("short_expiry",)


def test_extreme_moneyness_is_rejected(tmp_path: Path) -> None:
    quotes = _heston_quotes(
        {
            0: {
                "strike": 250.0,
                "mid": 0.5,
                "bid": 0.4,
                "ask": 0.6,
            }
        }
    )

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert len(prepared.selected_quotes) == 2
    assert prepared.stats.rejection_counts == {"extreme_moneyness": 1}
    assert prepared.rejected_quotes.loc[0, "reject_reasons"] == ("extreme_moneyness",)


def test_empty_selected_universe_returns_empty_status(tmp_path: Path) -> None:
    quotes = _heston_quotes(
        {
            0: {"expiry_years": 1.0 / 365.0},
            1: {"expiry_years": 2.0 / 365.0},
            2: {"expiry_years": 3.0 / 365.0},
        }
    )

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert prepared.status == "empty"
    assert prepared.selected_quotes.empty
    assert len(prepared.rejected_quotes) == 3
    assert prepared.quote_set is None
    assert prepared.preflight is None
    assert prepared.stats.selected_quote_count == 0
    assert prepared.stats.rejected_quote_count == 3
    assert prepared.stats.min_expiry_days is None
    assert prepared.stats.max_expiry_days is None


def test_preflight_blocked_universe_returns_blocked_unless_raise_on_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked_preflight(
        quotes: Any,
        *,
        raise_on_block: bool = False,
    ) -> HestonQuotePreflight:
        del raise_on_block
        return HestonQuotePreflight(
            quote_count=int(quotes.n_quotes),
            price_bound_violation_count=1,
            mid_outside_bid_ask_count=0,
            recommendation="block",
            messages=("forced preflight block",),
        )

    monkeypatch.setattr(model_ready, "preflight_heston_quotes", _blocked_preflight)

    prepared = model_ready.prepare_heston_market_fit(
        _bundle(tmp_path, _heston_quotes()),
    )

    assert prepared.status == "blocked"
    assert prepared.preflight is not None
    assert prepared.preflight.recommendation == "block"
    assert prepared.warnings == ("forced preflight block",)
    with pytest.raises(ValueError, match="forced preflight block"):
        model_ready.prepare_heston_market_fit(
            _bundle(tmp_path, _heston_quotes()),
            raise_on_block=True,
        )


def test_rejected_quotes_include_explicit_reasons_and_counts_match(
    tmp_path: Path,
) -> None:
    quotes = _heston_quotes(
        {
            0: {
                "expiry_years": 3.0 / 365.0,
                "strike": 250.0,
                "mid": 0.5,
                "bid": 0.4,
                "ask": 0.6,
            },
            1: {"iv": pd.NA},
        }
    )

    prepared = prepare_heston_market_fit(_bundle(tmp_path, quotes))

    assert "reject_reasons" in prepared.rejected_quotes.columns
    assert all(
        isinstance(value, tuple) for value in prepared.rejected_quotes["reject_reasons"]
    )
    assert all(prepared.rejected_quotes["reject_reasons"])
    flattened = _flatten_reasons(prepared.rejected_quotes)
    assert prepared.stats.rejection_counts == {
        reason: flattened.count(reason) for reason in dict.fromkeys(flattened)
    }
    assert prepared.stats.rejection_counts == {
        "short_expiry": 1,
        "missing_iv_for_seed": 1,
        "extreme_moneyness": 1,
    }


def test_input_heston_quotes_is_not_mutated(tmp_path: Path) -> None:
    quotes = _heston_quotes({0: {"iv": pd.NA}})
    original = quotes.copy(deep=True)

    prepare_heston_market_fit(_bundle(tmp_path, quotes))

    pd.testing.assert_frame_equal(quotes, original)
    assert "reject_reasons" not in quotes.columns


def test_missing_required_heston_column_raises_clear_validation_error(
    tmp_path: Path,
) -> None:
    quotes = _heston_quotes(drop_columns=("mid",))

    with pytest.raises(ValueError, match="missing required columns.*'mid'"):
        prepare_heston_market_fit(_bundle(tmp_path, quotes))
