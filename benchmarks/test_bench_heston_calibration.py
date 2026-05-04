from __future__ import annotations

import pytest

from option_pricing.diagnostics.heston.calibration import (
    _build_synthetic_heston_quotes,
    _default_quad_cfg,
    _default_seed_params,
    _default_true_params,
    _run_single_calibration,
)
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams


@pytest.fixture(scope="module")
def heston_calibration_smoke_setup() -> tuple[HestonObjective, HestonParams]:
    # NOTE: Grid and quadrature settings are smoke-benchmark choices, sized
    # for deterministic pytest-benchmark collection rather than production runs.
    quad_cfg = _default_quad_cfg()
    quotes = _build_synthetic_heston_quotes(
        market=None,
        true_params=_default_true_params(),
        expiries=None,
        log_moneyness=None,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
        random_seed=42,
        noise_vol_bps=0.0,
    )
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )
    return objective, _default_seed_params()


def test_bench_heston_calibration_analytic_jacobian(
    benchmark,
    heston_calibration_smoke_setup: tuple[HestonObjective, HestonParams],
) -> None:
    objective, seed_params = heston_calibration_smoke_setup
    benchmark(
        _run_single_calibration,
        objective=objective,
        seed_params=seed_params,
        mode="analytic",
        use_analytic_jac=True,
        repeat_index=0,
        warmup=False,
        loss="linear",
        method="trf",
        x_scale=None,
    )


def test_bench_heston_calibration_finite_difference_jacobian(
    benchmark,
    heston_calibration_smoke_setup: tuple[HestonObjective, HestonParams],
) -> None:
    objective, seed_params = heston_calibration_smoke_setup
    benchmark(
        _run_single_calibration,
        objective=objective,
        seed_params=seed_params,
        mode="finite_difference",
        use_analytic_jac=False,
        repeat_index=0,
        warmup=False,
        loss="linear",
        method="trf",
        x_scale=None,
    )
