from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston import (
    MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL,
    HestonModelComparisonDiagnostics,
    build_market_like_heston_quote_set,
    build_synthetic_heston_quote_set,
    run_heston_vs_local_vol_comparison,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.vol.ssvi import ESSVICalibrationConfig, ESSVIProjectionConfig


def _true_params() -> HestonParams:
    return HestonParams(kappa=1.5, vbar=0.04, eta=0.45, rho=-0.50, v=0.042)


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=45.0, n_panels=5, nodes_per_panel=5)


def _projection_cfg() -> ESSVIProjectionConfig:
    return ESSVIProjectionConfig(
        validation_nt=9,
        validation_y_min=-0.40,
        validation_y_max=0.40,
        validation_ny=21,
        dupire_nt=7,
        dupire_y_min=-0.35,
        dupire_y_max=0.35,
        dupire_ny=17,
        strict_validation=False,
    )


def test_market_like_heston_fixture_is_deterministic_and_sane() -> None:
    quotes = build_market_like_heston_quote_set()

    assert quotes.metadata is not None
    assert quotes.metadata["fixture_label"] == MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL
    assert quotes.metadata["generated_from_heston"] is False
    assert np.all(np.isfinite(quotes.strike))
    assert np.all(np.isfinite(quotes.expiry))
    assert np.all(np.isfinite(quotes.mid))
    assert quotes.iv_mid is not None
    assert np.all(np.isfinite(quotes.iv_mid))
    assert np.all(quotes.strike > 0.0)
    assert np.all(quotes.expiry > 0.0)
    assert np.all((quotes.iv_mid > 0.10) & (quotes.iv_mid < 0.40))

    quote_keys = {
        (float(T), float(K), bool(is_call))
        for T, K, is_call in zip(
            quotes.expiry,
            quotes.strike,
            quotes.is_call,
            strict=True,
        )
    }
    assert len(quote_keys) == quotes.n_quotes

    for expiry in np.unique(quotes.expiry):
        idx = np.flatnonzero(np.isclose(quotes.expiry, expiry))
        ordered = idx[np.argsort(quotes.log_moneyness[idx])]
        slice_prices = quotes.mid[ordered]
        slice_iv = quotes.iv_mid[ordered]
        assert np.all(np.diff(slice_prices) < 0.0)
        assert float(slice_iv[0]) > float(slice_iv[-1])


def test_heston_vs_local_vol_comparison_runs_direct_pde_on_market_like_fixture() -> (
    None
):
    quotes = build_market_like_heston_quote_set(
        expiries=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        log_moneyness=np.array([-0.10, 0.0, 0.10], dtype=np.float64),
    )

    report = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=_true_params(),
        heston_quad_cfg=_quad_cfg(),
        essvi_cfg=ESSVICalibrationConfig(max_nfev=300),
        essvi_projection_cfg=_projection_cfg(),
        local_vol_pde_max_quotes=3,
        local_vol_pde_Nx=31,
        local_vol_pde_Nt=41,
    )

    assert isinstance(report, HestonModelComparisonDiagnostics)
    assert set(report.tables) >= {
        "fit_errors",
        "error_summary",
        "direct_local_vol_pde_matched_error_summary",
        "tradeoff_summary",
        "held_out_comparison",
    }

    fit_errors = report.tables["fit_errors"]
    assert set(fit_errors["model"]) == {
        "Heston",
        "ESSVI local-vol proxy",
        "Direct local-vol PDE",
    }
    assert set(fit_errors["moneyness_bucket"]) == {
        "atm",
        "downside_wing",
        "upside_wing",
    }

    summary = report.tables["error_summary"]
    assert set(summary["model"]) == {
        "Heston",
        "ESSVI local-vol proxy",
        "Direct local-vol PDE",
    }
    assert {"atm", "downside_wing", "upside_wing"} <= set(summary["bucket"])
    metric_columns = [
        "price_rmse",
        "price_mae",
        "price_max_abs",
        "iv_rmse_bps",
        "iv_mae_bps",
        "iv_max_abs_bps",
    ]
    for column in metric_columns:
        values = summary[column].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(values))

    matched = report.tables["direct_local_vol_pde_matched_error_summary"]
    assert set(matched["model"]) == {
        "Heston",
        "ESSVI local-vol proxy",
        "Direct local-vol PDE",
    }
    assert {"all", "atm", "downside_wing", "upside_wing"} <= set(matched["bucket"])
    for _, bucket_rows in matched.groupby("bucket", sort=False):
        counts = bucket_rows.set_index("model")["n_quotes"]
        assert int(counts.loc["Heston"]) == int(counts.loc["ESSVI local-vol proxy"])
        assert int(counts.loc["Heston"]) == int(counts.loc["Direct local-vol PDE"])

    notes = " ".join(str(note) for note in report.meta["notes"])
    assert ("RE" + "VIEW:") not in notes
    assert "Dupire/PDE repricing audit" in notes
    assert "direct_local_vol_pde_matched_error_summary" in notes
    assert report.meta["local_vol_proxy_kind"] == "essvi_nodal_implied_surface"
    assert report.meta["direct_local_vol_pde_repricing"] is True
    assert report.meta["direct_local_vol_pde_success_count"] == 3
    assert (
        report.meta["comparison_fixture_label"] == MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL
    )
    assert report.meta["comparison_target"] == (
        "same HestonQuoteSet quotes repriced by both models"
    )

    direct = report.tables["direct_local_vol_pde"]
    assert len(direct) == 3
    assert set(direct["pde_status"]) == {"ok"}
    assert {
        "heston_price",
        "heston_iv",
        "target_iv",
        "local_vol_pde_price",
        "local_vol_pde_iv",
        "local_vol_pde_iv_residual_bps",
        "Nx",
        "Nt",
        "surface_source",
    } <= set(direct.columns)
    assert np.all(np.isfinite(direct["local_vol_pde_price"].to_numpy(np.float64)))
    assert np.all(np.isfinite(direct["local_vol_pde_iv"].to_numpy(np.float64)))


def test_heston_vs_local_vol_held_out_comparison_is_partitioned() -> None:
    quotes = build_synthetic_heston_quote_set(
        market=None,
        true_params=_true_params(),
        expiries=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        log_moneyness=np.array([-0.08, 0.0, 0.08], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        random_seed=321,
        noise_vol_bps=0.0,
    )
    held_out = np.zeros(quotes.n_quotes, dtype=np.bool_)
    held_out[2::3] = True

    report = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=_true_params(),
        held_out_mask=held_out,
        heston_quad_cfg=_quad_cfg(),
        essvi_cfg=ESSVICalibrationConfig(max_nfev=300),
        run_direct_local_vol_pde=False,
    )

    held_out_comparison = report.tables["held_out_comparison"]
    assert set(held_out_comparison["model"]) == {"Heston", "ESSVI local-vol proxy"}
    assert set(held_out_comparison["sample"]) == {"train", "held_out"}

    expected_train_count = int(np.count_nonzero(~held_out))
    expected_held_out_count = int(np.count_nonzero(held_out))
    for model_name in ("Heston", "ESSVI local-vol proxy"):
        model_rows = held_out_comparison.loc[
            held_out_comparison["model"] == model_name
        ].set_index("sample")
        assert int(model_rows.loc["train", "n_quotes"]) == expected_train_count
        assert int(model_rows.loc["held_out", "n_quotes"]) == expected_held_out_count

        fit_rows = (
            report.tables["fit_errors"]
            .loc[report.tables["fit_errors"]["model"] == model_name]
            .sort_values("quote_index")
        )
        assert (
            fit_rows["sample"].tolist()
            == np.where(
                held_out,
                "held_out",
                "train",
            ).tolist()
        )

    assert report.meta["held_out_mask_provided"] is True
    assert report.meta["train_quote_count"] == expected_train_count
    assert report.meta["held_out_quote_count"] == expected_held_out_count
    assert report.meta["sample_labels"] == ["train", "held_out"]


def test_heston_vs_local_vol_without_held_out_does_not_run_partition() -> None:
    quotes = build_synthetic_heston_quote_set(
        market=None,
        true_params=_true_params(),
        expiries=np.array([0.5, 1.0], dtype=np.float64),
        log_moneyness=np.array([-0.08, 0.0, 0.08], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        random_seed=111,
        noise_vol_bps=0.0,
    )

    report = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=_true_params(),
        heston_quad_cfg=_quad_cfg(),
        essvi_cfg=ESSVICalibrationConfig(max_nfev=300),
        run_direct_local_vol_pde=False,
    )

    assert report.tables["held_out_comparison"].empty
    assert report.meta["held_out_mask_provided"] is False
    assert report.meta["train_quote_count"] == quotes.n_quotes
    assert report.meta["held_out_quote_count"] == 0
    assert report.meta["sample_labels"] == ["fit"]
    assert set(report.tables["fit_errors"]["sample"]) == {"fit"}
