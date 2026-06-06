from __future__ import annotations

import numpy as np

from option_pricing.vol import LocalVolSurface
from option_pricing.vol.local_vol_gatheral import gatheral_local_var_diagnostics
from option_pricing.vol.ssvi import (
    ESSVINodalSurface,
    ESSVINodeSet,
    ESSVIProjectionConfig,
    project_essvi_nodes,
    validate_essvi_nodes,
)


def _sample_nodes() -> ESSVINodeSet:
    return ESSVINodeSet(
        expiries=np.array([0.25, 0.5, 1.0, 1.5], dtype=np.float64),
        theta=np.array([0.035, 0.048, 0.071, 0.092], dtype=np.float64),
        psi=np.array([0.16, 0.18, 0.205, 0.225], dtype=np.float64),
        rho=np.array([-0.20, -0.12, -0.04, 0.05], dtype=np.float64),
    )


def _tiny_calendar_violation_nodes() -> ESSVINodeSet:
    return ESSVINodeSet(
        expiries=np.array([0.12, 0.25, 0.5, 1.0], dtype=np.float64),
        theta=np.array(
            [0.062581545225, 0.07821334755, 0.1026294267, 0.11110671675],
            dtype=np.float64,
        ),
        psi=np.array(
            [0.3106086593, 0.33169569492500006, 0.420605727575, 0.446481536],
            dtype=np.float64,
        ),
        rho=np.array(
            [
                -0.5836967185749999,
                -0.5750925733,
                -0.36715948210000005,
                -0.30294336169999997,
            ],
            dtype=np.float64,
        ),
    )


def _dupire_invalid_nodes() -> ESSVINodeSet:
    return ESSVINodeSet(
        expiries=np.array([0.12, 0.25, 0.5, 1.0], dtype=np.float64),
        theta=np.array(
            [0.02758654, 0.05263399, 0.07005281, 0.088064], dtype=np.float64
        ),
        psi=np.array(
            [0.11930564, 0.22499516, 0.29403884, 0.35535321], dtype=np.float64
        ),
        rho=np.array([0.22555145, -0.3494312, -0.04119879, 0.12632], dtype=np.float64),
    )


def test_projected_surface_supports_dupire_local_vol_queries() -> None:
    nodes = _sample_nodes()
    projection = project_essvi_nodes(nodes)

    assert projection.success
    assert projection.surface is not None
    assert projection.candidate_surface is projection.surface
    assert projection.diag.dupire_invalid_count == 0

    lv = LocalVolSurface.from_implied(
        projection.surface,
        forward=lambda _T: 100.0,
    )
    out = lv.local_var(np.array([85.0, 100.0, 115.0], dtype=np.float64), 0.9)

    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_exact_nodal_surface_remains_available_as_fallback() -> None:
    nodes = _sample_nodes()
    projection = project_essvi_nodes(nodes)
    fallback = projection.fallback_surface

    assert isinstance(fallback, ESSVINodalSurface)
    assert np.all(
        np.isfinite(fallback.w(np.array([-0.3, 0.0, 0.3], dtype=np.float64), 0.8))
    )


def test_projected_surface_matches_nodal_short_end_extension() -> None:
    nodes = _sample_nodes()
    projection = project_essvi_nodes(nodes)

    assert projection.success
    assert projection.surface is not None

    T_short = 0.10
    y = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
    smooth_w = projection.surface.w(y, T_short)
    nodal_w = projection.fallback_surface.w(y, T_short)

    assert np.allclose(smooth_w, nodal_w, rtol=1.0e-8, atol=1.0e-10)

    lv = LocalVolSurface.from_implied(
        projection.surface,
        forward=lambda _T: 100.0,
    )
    sigma = lv.local_vol(np.array([100.0], dtype=np.float64), T_short)

    assert np.all(np.isfinite(sigma))
    assert float(sigma[0]) > 0.0


def test_calendar_only_rejection_exposes_candidate_surface() -> None:
    nodes = _tiny_calendar_violation_nodes()
    node_report = validate_essvi_nodes(nodes)

    assert node_report.ok

    projection = project_essvi_nodes(nodes)

    assert not projection.success
    assert projection.surface is None
    assert projection.params is None
    assert projection.candidate_surface is not None
    assert projection.candidate_params is not None
    assert projection.diag.validation is not None
    assert not projection.diag.calendar_ok
    assert projection.diag.calendar_bad_pair_count > 0
    assert 1.0e-8 < projection.diag.calendar_max_violation < 1.0e-7
    assert projection.diag.dupire_invalid_count == 0
    assert projection.diag.dupire_invalid_rate == 0.0
    assert "calendar=BAD" in projection.diag.static_noarb_message
    assert projection.diag.continuous_constraint_message == "OK"


def test_candidate_surface_localizes_bad_dupire_points() -> None:
    nodes = _dupire_invalid_nodes()
    projection = project_essvi_nodes(nodes)

    assert not projection.success
    assert projection.surface is None
    assert projection.candidate_surface is not None
    assert projection.diag.dupire_invalid_count > 0
    assert projection.diag.dupire_invalid_rate > 0.0

    invalid_points: list[tuple[float, float]] = []
    for tau in projection.diag.dupire_expiries:
        y = np.asarray(projection.diag.dupire_y_grid, dtype=np.float64)
        w, w_y, w_yy, w_T = projection.candidate_surface.w_and_derivs(y, float(tau))
        dupire = gatheral_local_var_diagnostics(
            y=y,
            w=w,
            w_y=w_y,
            w_yy=w_yy,
            w_T=w_T,
        )
        invalid_points.extend(
            (float(tau), float(y_i))
            for y_i in y[np.asarray(dupire.invalid, dtype=bool)]
        )

    assert len(invalid_points) == projection.diag.dupire_invalid_count
    assert all(np.isfinite(tau) and np.isfinite(y) for tau, y in invalid_points)


def test_relaxed_calendar_tolerance_certifies_tiny_calendar_rejection() -> None:
    nodes = _tiny_calendar_violation_nodes()

    strict_projection = project_essvi_nodes(nodes)
    relaxed_projection = project_essvi_nodes(
        nodes,
        cfg=ESSVIProjectionConfig(calendar_tol=1.0e-7),
    )

    assert not strict_projection.success
    assert strict_projection.diag.dupire_invalid_count == 0
    assert relaxed_projection.success
    assert relaxed_projection.surface is not None
    assert relaxed_projection.params is not None
    assert relaxed_projection.candidate_surface is relaxed_projection.surface
    assert relaxed_projection.diag.calendar_ok
    assert relaxed_projection.diag.calendar_bad_pair_count == 0


def test_projection_config_can_use_observed_quote_support() -> None:
    y_observed = np.array([-0.31, -0.08, 0.15, 0.42], dtype=np.float64)

    cfg = ESSVIProjectionConfig.from_observed_y(
        y_observed,
        validation_padding=0.03,
        dupire_padding=0.01,
    )

    np.testing.assert_allclose(
        [
            cfg.validation_y_min,
            cfg.validation_y_max,
            cfg.dupire_y_min,
            cfg.dupire_y_max,
        ],
        [-0.34, 0.45, -0.32, 0.43],
        atol=1.0e-15,
    )
