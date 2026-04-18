from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning

import option_pricing.models.heston.fourier as heston_fourier
from option_pricing.models.heston import (
    HestonParams,
    P_j_Scalar,
    recommend_heston_quadrature_config,
)
from option_pricing.models.heston.fourier import (
    HestonIntegralDiagnostics,
    P_j_with_diagnostics,
)
from option_pricing.numerics.quadrature import (
    QuadratureConfig,
    build_gauss_legendre_rule,
)


def _sample_params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _sample_rule():
    cfg = QuadratureConfig(u_max=10.0, n_panels=2, nodes_per_panel=4)
    return build_gauss_legendre_rule(cfg)


def _patch_quad_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
    **quad_kwargs: float | int,
) -> None:
    original_quad = heston_fourier.quad

    def _quad_with_overrides(func, a, b, args=(), complex_func=False):
        return original_quad(
            func,
            a=a,
            b=b,
            args=args,
            complex_func=complex_func,
            **quad_kwargs,
        )

    monkeypatch.setattr(heston_fourier, "quad", _quad_with_overrides)


def _gauss_reference_probability() -> float:
    return P_j_Scalar(
        x=-0.2,
        tau=1.0,
        params=_sample_params(),
        j=0,
        backend="gauss_legendre",
        quad_cfg=QuadratureConfig(u_max=240.0, n_panels=48, nodes_per_panel=32),
    )


def _gauss_errors(configs: list[QuadratureConfig]) -> np.ndarray:
    params = _sample_params()
    reference = _gauss_reference_probability()
    return np.asarray(
        [
            abs(
                P_j_Scalar(
                    x=-0.2,
                    tau=1.0,
                    params=params,
                    j=0,
                    backend="gauss_legendre",
                    quad_cfg=cfg,
                )
                - reference
            )
            for cfg in configs
        ],
        dtype=float,
    )


def _format_convergence_diagnostics(
    *,
    label: str,
    values: list[float | int],
    errors: np.ndarray,
) -> str:
    rows = [f"refinement={label}", "value | abs_error_to_reference"]
    for value, error in zip(values, errors, strict=True):
        rows.append(f"{value!s:>5} | {float(error):.12e}")
    return "\n".join(rows)


def _recommended_gauss_quad_cfg(
    *, x: float, tau: float, params: HestonParams
) -> QuadratureConfig:
    return recommend_heston_quadrature_config(
        x=x,
        tau=tau,
        params=params,
        quality="robust",
    )


def _backend_probabilities(
    *, params: HestonParams, tau: float, backend: str
) -> np.ndarray:
    probabilities: list[float] = []
    for x in (-0.2, 0.0, 0.2):
        quad_cfg = (
            _recommended_gauss_quad_cfg(x=float(x), tau=tau, params=params)
            if backend == "gauss_legendre"
            else None
        )
        for j in (0, 1):
            probabilities.append(
                P_j_Scalar(
                    x=float(x),
                    tau=tau,
                    params=params,
                    j=j,
                    backend=backend,
                    quad_cfg=quad_cfg,
                )
            )
    return np.asarray(probabilities, dtype=float)


def _format_backend_comparison(
    *,
    regime_name: str,
    quad_values: np.ndarray,
    gauss_values: np.ndarray,
) -> str:
    diffs = gauss_values - quad_values
    rows = [
        f"regime={regime_name}",
        ("max_abs_diff=" f"{float(np.max(np.abs(diffs))):.12e}"),
        "",
        "index | quad | gauss | diff",
    ]
    for index, (quad_value, gauss_value, diff) in enumerate(
        zip(quad_values, gauss_values, diffs, strict=True)
    ):
        rows.append(
            f"{index:5d} | {float(quad_value): .12f} |"
            f" {float(gauss_value): .12f} | {float(diff): .12e}"
        )
    return "\n".join(rows)


def test_p_j_dispatches_to_quad_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _sample_params()
    calls: dict[str, object] = {}

    def _fake_integrate_quad(x: float, tau: float, params_: HestonParams, j: int):
        calls["quad"] = (x, tau, params_, j)
        return 0.75, 0.125

    def _fail_resolve_gauss_rule(*, quad_cfg, rule):
        raise AssertionError("quad backend should not resolve a Gauss-Legendre rule")

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fake_integrate_quad)
    monkeypatch.setattr(heston_fourier, "_resolve_gauss_rule", _fail_resolve_gauss_rule)

    probability = P_j_Scalar(x=0.2, tau=1.0, params=params, j=1, backend="quad")

    assert calls["quad"] == (0.2, 1.0, params, 1)
    assert probability == pytest.approx(0.5 + 0.75 / np.pi)


def test_p_j_dispatches_to_gauss_legendre_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _sample_params()
    quad_cfg = QuadratureConfig(u_max=50.0, n_panels=8, nodes_per_panel=4)
    sentinel_rule = object()
    calls: dict[str, object] = {}

    def _fail_quad_backend(*args, **kwargs):
        raise AssertionError("Gauss-Legendre backend should not call quad")

    def _fake_resolve_gauss_rule(*, quad_cfg, rule):
        calls["resolve"] = (quad_cfg, rule)
        return sentinel_rule

    def _fake_integrate_fixed_rule(
        x: float,
        tau: float,
        params_: HestonParams,
        j: int,
        rule: object,
    ) -> float:
        calls["gauss"] = (x, tau, params_, j, rule)
        return 0.25

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fail_quad_backend)
    monkeypatch.setattr(heston_fourier, "_resolve_gauss_rule", _fake_resolve_gauss_rule)
    monkeypatch.setattr(
        heston_fourier,
        "_integrate_pj_fixed_rule",
        _fake_integrate_fixed_rule,
    )

    probability = P_j_Scalar(
        x=-0.1,
        tau=0.75,
        params=params,
        j=0,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    assert calls["resolve"] == (quad_cfg, None)
    assert calls["gauss"] == (-0.1, 0.75, params, 0, sentinel_rule)
    assert probability == pytest.approx(0.5 + 0.25 / np.pi)


def test_p_j_dispatches_to_gauss_legendre_backend_with_explicit_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _sample_params()
    rule = _sample_rule()
    calls: dict[str, object] = {}

    def _fail_quad_backend(*args, **kwargs):
        raise AssertionError("Gauss-Legendre backend should not call quad")

    def _fake_resolve_gauss_rule(*, quad_cfg, rule):
        calls["resolve"] = (quad_cfg, rule)
        return rule

    def _fake_integrate_fixed_rule(
        x: float,
        tau: float,
        params_: HestonParams,
        j: int,
        rule,
    ) -> float:
        calls["gauss"] = (x, tau, params_, j, rule)
        return 0.125

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fail_quad_backend)
    monkeypatch.setattr(heston_fourier, "_resolve_gauss_rule", _fake_resolve_gauss_rule)
    monkeypatch.setattr(
        heston_fourier,
        "_integrate_pj_fixed_rule",
        _fake_integrate_fixed_rule,
    )

    probability = P_j_Scalar(
        x=0.15,
        tau=0.5,
        params=params,
        j=1,
        backend="gauss_legendre",
        rule=rule,
    )

    assert calls["resolve"] == (None, rule)
    assert calls["gauss"] == (0.15, 0.5, params, 1, rule)
    assert probability == pytest.approx(0.5 + 0.125 / np.pi)


def test_p_j_with_diagnostics_dispatches_to_quad_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _sample_params()
    calls: dict[str, object] = {}

    def _fake_integrate_quad(x: float, tau: float, params_: HestonParams, j: int):
        calls["quad"] = (x, tau, params_, j)
        return 1.25, 1e-7

    def _fail_resolve_gauss_rule(*, quad_cfg, rule):
        raise AssertionError("quad backend should not resolve a Gauss-Legendre rule")

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fake_integrate_quad)
    monkeypatch.setattr(heston_fourier, "_resolve_gauss_rule", _fail_resolve_gauss_rule)

    diagnostics = P_j_with_diagnostics(
        x=0.0,
        tau=0.5,
        params=params,
        j=0,
        backend="quad",
    )

    assert calls["quad"] == (0.0, 0.5, params, 0)
    assert diagnostics.backend == "quad"
    assert diagnostics.j == 0
    assert diagnostics.x == 0.0
    assert diagnostics.tau == 0.5
    assert diagnostics.total_integral == pytest.approx(1.25)
    assert diagnostics.probability == pytest.approx(0.5 + 1.25 / np.pi)
    assert diagnostics.panel_contribs is None
    assert diagnostics.panel_edges is None
    assert diagnostics.quad_error_estimate == pytest.approx(1e-7)


def test_p_j_with_diagnostics_dispatches_to_gauss_legendre_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _sample_params()
    quad_cfg = QuadratureConfig(u_max=40.0, n_panels=6, nodes_per_panel=4)
    sentinel_rule = object()
    expected = HestonIntegralDiagnostics(
        backend="gauss_legendre",
        j=1,
        x=0.1,
        tau=0.9,
        total_integral=0.8,
        probability=0.5 + 0.8 / np.pi,
        panel_contribs=np.array([0.2, 0.3, 0.3], dtype=float),
        panel_edges=np.array([0.0, 10.0, 20.0, 30.0], dtype=float),
        quad_error_estimate=None,
    )
    calls: dict[str, object] = {}

    def _fail_quad_backend(*args, **kwargs):
        raise AssertionError("Gauss-Legendre backend should not call quad")

    def _fake_resolve_gauss_rule(*, quad_cfg, rule):
        calls["resolve"] = (quad_cfg, rule)
        return sentinel_rule

    def _fake_integrate_fixed_rule_with_diagnostics(
        x: float,
        tau: float,
        params: HestonParams,
        j: int,
        rule: object,
    ) -> HestonIntegralDiagnostics:
        calls["gauss"] = (x, tau, params, j, rule)
        return expected

    monkeypatch.setattr(heston_fourier, "_integrate_pj_quad", _fail_quad_backend)
    monkeypatch.setattr(heston_fourier, "_resolve_gauss_rule", _fake_resolve_gauss_rule)
    monkeypatch.setattr(
        heston_fourier,
        "_integrate_pj_fixed_rule_with_diagnostics",
        _fake_integrate_fixed_rule_with_diagnostics,
    )

    diagnostics = P_j_with_diagnostics(
        x=0.1,
        tau=0.9,
        params=params,
        j=1,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    assert calls["resolve"] == (quad_cfg, None)
    assert calls["gauss"] == (0.1, 0.9, params, 1, sentinel_rule)
    assert diagnostics is expected


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(P_j_Scalar, id="P_j_Scalar"),
        pytest.param(P_j_with_diagnostics, id="P_j_with_diagnostics"),
    ],
)
def test_gauss_backend_rejects_quad_cfg_and_rule_together(fn) -> None:
    quad_cfg = QuadratureConfig(u_max=50.0, n_panels=8, nodes_per_panel=4)
    rule = _sample_rule()

    with pytest.raises(ValueError, match="Pass either quad_cfg or rule, not both"):
        fn(
            x=0.0,
            tau=1.0,
            params=_sample_params(),
            j=0,
            backend="gauss_legendre",
            quad_cfg=quad_cfg,
            rule=rule,
        )


def test_heston_build_gauss_rule_reuses_generic_cached_rule() -> None:
    build_gauss_legendre_rule.cache_clear()
    cfg = QuadratureConfig(u_max=10.0, n_panels=2, nodes_per_panel=4)

    rule = heston_fourier._build_heston_gauss_rule(cfg)

    assert rule is build_gauss_legendre_rule(cfg)


def test_gauss_backend_converges_as_u_max_increases() -> None:
    u_max_values = [40.0, 80.0, 120.0, 180.0]
    errors = _gauss_errors(
        [
            QuadratureConfig(u_max=u_max, n_panels=24, nodes_per_panel=16)
            for u_max in u_max_values
        ]
    )

    assert np.all(np.isfinite(errors))
    assert errors[-1] <= errors[0], _format_convergence_diagnostics(
        label="u_max",
        values=u_max_values,
        errors=errors,
    )
    assert errors[-1] <= 1e-10, _format_convergence_diagnostics(
        label="u_max",
        values=u_max_values,
        errors=errors,
    )


def test_gauss_backend_converges_as_n_panels_increase() -> None:
    panel_counts = [6, 12, 24, 36]
    errors = _gauss_errors(
        [
            QuadratureConfig(u_max=180.0, n_panels=n_panels, nodes_per_panel=16)
            for n_panels in panel_counts
        ]
    )

    assert np.all(np.isfinite(errors))
    assert errors[-1] <= errors[0], _format_convergence_diagnostics(
        label="n_panels",
        values=panel_counts,
        errors=errors,
    )
    assert errors[-1] <= 1e-10, _format_convergence_diagnostics(
        label="n_panels",
        values=panel_counts,
        errors=errors,
    )


def test_gauss_backend_converges_as_nodes_per_panel_increase() -> None:
    node_counts = [4, 8, 16, 24]
    errors = _gauss_errors(
        [
            QuadratureConfig(u_max=180.0, n_panels=24, nodes_per_panel=n_nodes)
            for n_nodes in node_counts
        ]
    )

    assert np.all(np.isfinite(errors))
    assert errors[-1] <= errors[0], _format_convergence_diagnostics(
        label="nodes_per_panel",
        values=node_counts,
        errors=errors,
    )
    assert errors[-1] <= 1e-10, _format_convergence_diagnostics(
        label="nodes_per_panel",
        values=node_counts,
        errors=errors,
    )


@pytest.mark.parametrize(
    ("regime_name", "params", "tau", "tolerance"),
    [
        pytest.param(
            "ordinary",
            HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05),
            1.0,
            1e-10,
            id="ordinary",
        ),
        pytest.param(
            "near_black",
            HestonParams(kappa=3.0, vbar=0.04, eta=1e-4, rho=-0.35, v=0.07),
            1.0,
            1e-8,
            id="near-black",
        ),
        pytest.param(
            "near_deterministic",
            HestonParams(kappa=100.0, vbar=0.04, eta=1e-6, rho=0.0, v=0.04),
            1.0,
            1e-3,
            id="near-deterministic",
        ),
        pytest.param(
            "awkward_corr_volvol",
            HestonParams(kappa=0.8, vbar=0.09, eta=0.95, rho=0.75, v=0.03),
            0.75,
            1e-5,
            id="awkward-corr-volvol",
        ),
    ],
)
def test_gauss_legendre_tracks_quad_across_representative_regimes(
    monkeypatch: pytest.MonkeyPatch,
    regime_name: str,
    params: HestonParams,
    tau: float,
    tolerance: float,
) -> None:
    _patch_quad_with_overrides(
        monkeypatch,
        limit=300,
        epsabs=1e-12,
        epsrel=1e-12,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        quad_values = _backend_probabilities(
            params=params,
            tau=tau,
            backend="quad",
        )

    gauss_values = _backend_probabilities(
        params=params,
        tau=tau,
        backend="gauss_legendre",
    )
    max_abs_diff = float(np.max(np.abs(gauss_values - quad_values)))

    assert np.all(np.isfinite(quad_values))
    assert np.all(np.isfinite(gauss_values))
    assert max_abs_diff <= tolerance, _format_backend_comparison(
        regime_name=regime_name,
        quad_values=quad_values,
        gauss_values=gauss_values,
    )
