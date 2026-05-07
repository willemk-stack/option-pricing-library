from __future__ import annotations

import numpy as np
import pytest

import option_pricing.models.heston.fourier as heston_fourier
from option_pricing.models.heston.fourier import (
    HestonIntegralWarning,
    HestonPanelReason,
    _compute_fixed_rule_panel_reason,
    _compute_global_warning_flags,
    heston_probability_batch_with_diagnostics,
    heston_probability_with_diagnostics,
    heston_warning_calibration_action,
    recommend_heston_quadrature_config,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig


def _tail_warning_params() -> HestonParams:
    return HestonParams(
        kappa=0.4,
        vbar=0.04,
        eta=1.2,
        rho=-0.95,
        v=0.02,
    )


def _hard_cancellation_params() -> HestonParams:
    return HestonParams(
        kappa=0.1,
        vbar=0.04,
        eta=2.5,
        rho=-0.99,
        v=1e-4,
    )


def test_compute_global_warning_flags_probability_and_quad_error() -> None:
    flags, tail_abs_fraction, cancellation_ratio = _compute_global_warning_flags(
        total_integral=np.array([0.1, np.nan], dtype=np.float64),
        probability=np.array([1.0 + 1.0e-6, np.nan], dtype=np.float64),
        quad_error_estimate=np.array([1.0e-2, 0.0], dtype=np.float64),
    )

    assert tail_abs_fraction is None
    assert cancellation_ratio is None

    assert int(flags[0]) & int(HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE)
    assert int(flags[0]) & int(HestonIntegralWarning.QUAD_ERROR_LARGE)

    assert int(flags[1]) & int(HestonIntegralWarning.NONFINITE_TOTAL)
    assert int(flags[1]) & int(HestonIntegralWarning.NONFINITE_PROBABILITY)


def test_heston_warning_calibration_action_policy_mapping() -> None:
    assert (
        heston_warning_calibration_action(int(HestonIntegralWarning.NONFINITE_TOTAL))
        == "block"
    )
    assert (
        heston_warning_calibration_action(
            int(HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE)
        )
        == "quarantine"
    )
    assert (
        heston_warning_calibration_action(
            int(HestonIntegralWarning.LARGE_TAIL_FRACTION)
        )
        == "review"
    )
    assert heston_warning_calibration_action(0) == "ok"


def test_compute_fixed_rule_panel_reason_marks_nonfinite_tail_and_spike() -> None:
    values_panel = np.array(
        [[1.0, 1.0], [1.0, 1.0], [np.nan, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    panel_contribs = np.array([1.0, 20.0, np.nan, 18.0], dtype=np.float64)

    panel_reason = _compute_fixed_rule_panel_reason(values_panel, panel_contribs)

    assert panel_reason.shape == (4,)
    assert int(panel_reason[2]) & int(HestonPanelReason.NONFINITE_INTEGRAND)
    assert int(panel_reason[2]) & int(HestonPanelReason.NONFINITE_PANEL_CONTRIB)
    assert int(panel_reason[1]) & int(HestonPanelReason.OSCILLATION_SPIKE)
    assert int(panel_reason[3]) & int(HestonPanelReason.TAIL_TOO_LARGE)


def test_compute_fixed_rule_panel_reason_ignores_tiny_oscillation_spikes() -> None:
    values_panel = np.ones((7, 2), dtype=np.float64)
    panel_contribs = np.array(
        [0.3, 0.3, 1.0e-6, 1.0e-5, 1.0e-6, 0.3, 0.3],
        dtype=np.float64,
    )

    panel_reason = _compute_fixed_rule_panel_reason(values_panel, panel_contribs)

    assert panel_reason.shape == (7,)
    assert not (int(panel_reason[3]) & int(HestonPanelReason.OSCILLATION_SPIKE))


def test_compute_global_warning_flags_ignores_near_origin_only_panels() -> None:
    panel_invalid = np.array([True, True, True, True, True], dtype=np.bool_)
    panel_reason = np.array(
        [
            int(HestonPanelReason.UNDERRESOLVED_NEAR_ORIGIN),
            int(HestonPanelReason.UNDERRESOLVED_NEAR_ORIGIN),
            int(HestonPanelReason.OSCILLATION_SPIKE),
            int(HestonPanelReason.OSCILLATION_SPIKE),
            int(HestonPanelReason.OSCILLATION_SPIKE),
        ],
        dtype=np.uint32,
    )

    flags, _, _ = _compute_global_warning_flags(
        total_integral=np.array([10.0], dtype=np.float64),
        probability=np.array([0.5], dtype=np.float64),
        panel_invalid=panel_invalid,
        panel_contribs=np.ones(5, dtype=np.float64),
        panel_reason=panel_reason,
    )

    assert not (int(flags[0]) & int(HestonIntegralWarning.TOO_MANY_BAD_PANELS))


def test_compute_global_warning_flags_scales_with_panel_count() -> None:
    panel_invalid = np.zeros(60, dtype=np.bool_)
    panel_reason = np.zeros(60, dtype=np.uint32)
    panel_invalid[:5] = True
    panel_reason[:5] = int(HestonPanelReason.OSCILLATION_SPIKE)

    five_flags, _, _ = _compute_global_warning_flags(
        total_integral=np.array([100.0], dtype=np.float64),
        probability=np.array([0.5], dtype=np.float64),
        panel_invalid=panel_invalid,
        panel_contribs=np.ones(60, dtype=np.float64),
        panel_reason=panel_reason,
    )

    panel_invalid[5] = True
    panel_reason[5] = int(HestonPanelReason.OSCILLATION_SPIKE)
    six_flags, _, _ = _compute_global_warning_flags(
        total_integral=np.array([100.0], dtype=np.float64),
        probability=np.array([0.5], dtype=np.float64),
        panel_invalid=panel_invalid,
        panel_contribs=np.ones(60, dtype=np.float64),
        panel_reason=panel_reason,
    )

    assert not (int(five_flags[0]) & int(HestonIntegralWarning.TOO_MANY_BAD_PANELS))
    assert int(six_flags[0]) & int(HestonIntegralWarning.TOO_MANY_BAD_PANELS)


def test_scalar_fixed_rule_diagnostics_populate_new_fields() -> None:
    params = HestonParams(
        kappa=2.0,
        vbar=0.04,
        eta=0.6,
        rho=-0.7,
        v=0.04,
    )
    cfg = recommend_heston_quadrature_config(
        x=0.5,
        tau=0.5,
        params=params,
        quality="balanced",
    )

    diag = heston_probability_with_diagnostics(
        x=0.25,
        tau=0.5,
        params=params,
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=cfg,
    )

    assert diag.quad_cfg == cfg
    assert diag.panel_contribs is not None
    assert diag.panel_edges is not None
    assert diag.panel_invalid is not None
    assert diag.panel_reason is not None
    assert isinstance(diag.warning_flags, HestonIntegralWarning)
    assert diag.tail_abs_fraction is not None
    assert diag.cancellation_ratio is not None

    assert diag.panel_contribs.shape == (cfg.n_panels,)
    assert diag.panel_edges.shape == (cfg.n_panels + 1,)
    assert diag.panel_invalid.shape == (cfg.n_panels,)
    assert diag.panel_reason.shape == (cfg.n_panels,)


def test_batch_fixed_rule_diagnostics_shapes_match_x_shape() -> None:
    params = HestonParams(
        kappa=2.0,
        vbar=0.04,
        eta=0.6,
        rho=-0.7,
        v=0.04,
    )
    x = np.array([[-0.25, 0.0, 0.25], [0.5, 0.75, 1.0]], dtype=np.float64)

    cfg = recommend_heston_quadrature_config(
        x=float(np.max(np.abs(x))),
        tau=0.5,
        params=params,
        quality="balanced",
    )

    diag = heston_probability_batch_with_diagnostics(
        x=x,
        tau=0.5,
        params=params,
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=cfg,
    )

    assert diag.quad_cfg == cfg
    assert diag.total_integral.shape == x.shape
    assert diag.probability.shape == x.shape
    assert diag.warning_flags is not None
    assert diag.warning_flags.shape == x.shape

    assert diag.panel_contribs is not None
    assert diag.panel_invalid is not None
    assert diag.panel_reason is not None

    assert diag.panel_contribs.shape == x.shape + (cfg.n_panels,)
    assert diag.panel_invalid.shape == x.shape + (cfg.n_panels,)
    assert diag.panel_reason.shape == x.shape + (cfg.n_panels,)

    assert diag.tail_abs_fraction is not None
    assert diag.cancellation_ratio is not None
    assert diag.tail_abs_fraction.shape == x.shape
    assert diag.cancellation_ratio.shape == x.shape


def test_batch_quad_diagnostics_only_populate_global_fields() -> None:
    params = HestonParams(
        kappa=2.0,
        vbar=0.04,
        eta=0.6,
        rho=-0.7,
        v=0.04,
    )
    x = np.array([-0.25, 0.0, 0.25], dtype=np.float64)

    diag = heston_probability_batch_with_diagnostics(
        x=x,
        tau=0.5,
        params=params,
        probability_index=0,
        backend="quad",
    )

    assert diag.quad_cfg is None
    assert diag.panel_contribs is None
    assert diag.panel_edges is None
    assert diag.panel_invalid is None
    assert diag.panel_reason is None

    assert diag.warning_flags is not None
    assert diag.warning_flags.shape == x.shape
    assert diag.quad_error_estimate is not None
    assert diag.quad_error_estimate.shape == x.shape
    assert diag.tail_abs_fraction is None
    assert diag.cancellation_ratio is None


def test_scalar_quad_diagnostics_marks_nonfinite_total_and_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_integrate_pj_quad(
        x: float,
        tau: float,
        params: HestonParams,
        j: int,
    ) -> tuple[float, float]:
        return float("nan"), 0.0

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fake_integrate_pj_quad)

    diag = heston_probability_with_diagnostics(
        x=0.0,
        tau=0.5,
        params=_tail_warning_params(),
        probability_index=0,
        backend="quad",
    )

    assert np.isnan(diag.total_integral)
    assert np.isnan(diag.probability)
    assert int(diag.warning_flags) & int(HestonIntegralWarning.NONFINITE_TOTAL)
    assert int(diag.warning_flags) & int(HestonIntegralWarning.NONFINITE_PROBABILITY)


def test_scalar_quad_diagnostics_marks_out_of_range_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_integrate_pj_quad(
        x: float,
        tau: float,
        params: HestonParams,
        j: int,
    ) -> tuple[float, float]:
        return 2.0 * np.pi, 0.0

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fake_integrate_pj_quad)

    diag = heston_probability_with_diagnostics(
        x=0.0,
        tau=0.5,
        params=_tail_warning_params(),
        probability_index=1,
        backend="quad",
    )

    assert diag.probability > 1.0
    assert int(diag.warning_flags) & int(HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE)


def test_too_small_u_max_triggers_tail_warnings_on_hard_regime() -> None:
    diag = heston_probability_with_diagnostics(
        x=0.0,
        tau=0.1,
        params=_tail_warning_params(),
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=QuadratureConfig(u_max=10.0, n_panels=8, nodes_per_panel=8),
    )

    assert int(diag.warning_flags) & int(HestonIntegralWarning.LARGE_TAIL_FRACTION)
    assert not (
        int(diag.warning_flags) & int(HestonIntegralWarning.TOO_MANY_BAD_PANELS)
    )
    assert diag.tail_abs_fraction is not None
    assert diag.tail_abs_fraction > 0.5
    assert diag.panel_reason is not None
    assert np.all(
        (diag.panel_reason[-3:] & np.uint32(int(HestonPanelReason.UNDERRESOLVED_TAIL)))
        != 0
    )
    assert np.all(
        (diag.panel_reason[-3:] & np.uint32(int(HestonPanelReason.TAIL_TOO_LARGE))) != 0
    )


def test_hard_regime_exposes_large_cancellation_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(heston_fourier, "HESTON_DIAG_CANCELLATION_RATIO_WARN", 10.0)

    diag = heston_probability_with_diagnostics(
        x=0.0,
        tau=0.03,
        params=_hard_cancellation_params(),
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=QuadratureConfig(u_max=10.0, n_panels=16, nodes_per_panel=4),
    )

    assert diag.cancellation_ratio is not None
    assert diag.cancellation_ratio > 10.0
    assert int(diag.warning_flags) & int(HestonIntegralWarning.EXCESSIVE_CANCELLATION)


def test_gauss_batch_warning_flags_match_scalar_warning_flags() -> None:
    params = _tail_warning_params()
    x = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float64)
    cfg = QuadratureConfig(u_max=10.0, n_panels=8, nodes_per_panel=8)

    batch_diag = heston_probability_batch_with_diagnostics(
        x=x,
        tau=0.1,
        params=params,
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=cfg,
    )
    scalar_diags = [
        heston_probability_with_diagnostics(
            x=float(x_i),
            tau=0.1,
            params=params,
            probability_index=0,
            backend="gauss_legendre",
            quad_cfg=cfg,
        )
        for x_i in x
    ]

    expected_flags = np.asarray(
        [int(diag.warning_flags) for diag in scalar_diags],
        dtype=np.uint32,
    )
    expected_tail = np.asarray(
        [diag.tail_abs_fraction for diag in scalar_diags],
        dtype=np.float64,
    )
    expected_cancellation = np.asarray(
        [diag.cancellation_ratio for diag in scalar_diags],
        dtype=np.float64,
    )

    assert batch_diag.warning_flags is not None
    assert batch_diag.tail_abs_fraction is not None
    assert batch_diag.cancellation_ratio is not None

    np.testing.assert_array_equal(batch_diag.warning_flags, expected_flags)
    np.testing.assert_allclose(
        batch_diag.tail_abs_fraction,
        expected_tail,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        batch_diag.cancellation_ratio,
        expected_cancellation,
        atol=1e-12,
        rtol=0.0,
    )
