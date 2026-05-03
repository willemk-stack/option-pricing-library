"""Deterministic Heston calibration diagnostics and local-vol comparison demo."""

from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston import (
    build_synthetic_heston_quote_set,
    run_heston_calibration_fit_diagnostics,
    run_heston_vs_local_vol_comparison,
)
from option_pricing.models.heston.calibration import calibrate_heston_multistart
from option_pricing.models.heston.calibration.bounds import HestonCalibrationBounds
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.vol.ssvi import ESSVICalibrationConfig


def main() -> None:
    true_params = HestonParams(kappa=1.7, vbar=0.04, eta=0.55, rho=-0.55, v=0.045)
    quad_cfg = QuadratureConfig(u_max=50.0, n_panels=6, nodes_per_panel=6)
    quotes = build_synthetic_heston_quote_set(
        market=None,
        true_params=true_params,
        expiries=np.array([0.5, 1.0, 2.0], dtype=np.float64),
        log_moneyness=np.array([-0.12, -0.06, 0.0, 0.06, 0.12], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
        random_seed=123,
        noise_vol_bps=1.0,
    )

    held_out = np.zeros(quotes.n_quotes, dtype=np.bool_)
    held_out[4::5] = True

    bounds = HestonCalibrationBounds()
    multistart = calibrate_heston_multistart(
        quotes=quotes,
        objective_type="vega_scaled_price",
        bounds=bounds,
        quad_cfg=quad_cfg,
        max_seeds=6,
        parameter_transform="bounded",
        loss="linear",
        max_nfev=100,
    )

    fit_report = run_heston_calibration_fit_diagnostics(
        quotes=quotes,
        fit=multistart,
        true_params=true_params,
        held_out_mask=held_out,
        quad_cfg=quad_cfg,
    )
    comparison = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=multistart,
        held_out_mask=held_out,
        heston_quad_cfg=quad_cfg,
        essvi_cfg=ESSVICalibrationConfig(max_nfev=800),
    )

    print("Best Heston parameters:")
    print(multistart.best_params)
    print("\nHeld-out calibration errors:")
    print(fit_report.tables["held_out_errors"])
    print("\nModel comparison summary:")
    print(comparison.tables["error_summary"])
    print("\nTradeoff summary:")
    print(comparison.tables["tradeoff_summary"])


if __name__ == "__main__":
    main()
