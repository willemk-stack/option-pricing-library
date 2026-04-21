from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from option_pricing.diagnostics.heston import run_heston_pricing_diagnostics
from option_pricing.diagnostics.heston.contracts import HestonDiagnosticsBackend
from option_pricing.diagnostics.heston.models import HestonDiagnosticsReport
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.types import MarketData, OptionType

# Owner review required before treating this helper as frozen regression policy.
# These seeds, strike grids, and broad stress expectations are intentionally
# centralized here so Step 6 does not bury approval-gated choices in scattered
# asserts.


@dataclass(frozen=True, slots=True)
class Step6StressCase:
    label: str
    params: HestonParams
    tau: float
    review_note: str
    expect_any_non_ok_severity: bool = False
    expect_any_suspicious: bool = False


@dataclass(frozen=True, slots=True)
class FrozenAcceptanceSlice:
    label: str
    case_label: str
    review_note: str
    max_backend_discrepancy: float
    max_config_price_span: float
    max_suspicious_strikes: int
    max_non_ok_severity_count: int
    max_parameter_sensitivity_strikes: int
    require_any_review_signal: bool = False
    use_recommended_cfg: bool = True


STEP6_OWNER_APPROVAL_NOTE = (
    "Owner review required before freezing Step 6 acceptance semantics for "
    "these parameter sets, strike grids, perturbation bumps, and broad stress "
    "expectation flags."
)

STEP6_FROZEN_ACCEPTANCE_NOTE = (
    "Two frozen acceptance slices are enforced here: one ordinary slice that "
    "should stay mostly clean, and one stress slice where warnings are allowed "
    "but backend and config-sweep stability must remain bounded."
)

PROVISIONAL_MARKET = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)

PROVISIONAL_STRIKE_GRID = np.array(
    [80.0, 90.0, 100.0, 110.0, 120.0],
    dtype=np.float64,
)

PROVISIONAL_GAUSS_LEGENDRE_CFG = QuadratureConfig(
    u_max=120.0,
    n_panels=12,
    nodes_per_panel=12,
)

PROVISIONAL_STRESS_CASES: tuple[Step6StressCase, ...] = (
    Step6StressCase(
        label="normal_case",
        params=HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05),
        tau=0.75,
        review_note="Baseline acceptance seed for a readable notebook-friendly slice.",
    ),
    Step6StressCase(
        label="high_vol_of_vol",
        params=HestonParams(kappa=2.0, vbar=0.04, eta=1.10, rho=-0.70, v=0.05),
        tau=0.75,
        review_note="Expected to surface at least one stressed signal.",
        expect_any_suspicious=True,
    ),
    Step6StressCase(
        label="rho_near_minus_one",
        params=HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.98, v=0.05),
        tau=0.75,
        review_note="Near-degenerate correlation regime; review expectations manually.",
        expect_any_suspicious=True,
    ),
    Step6StressCase(
        label="short_maturity",
        params=HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05),
        tau=0.05,
        review_note="Short-dated slice where continuity and backend differences matter.",
        expect_any_non_ok_severity=True,
        expect_any_suspicious=True,
    ),
    Step6StressCase(
        label="long_maturity",
        params=HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05),
        tau=3.0,
        review_note="Long-dated slice for contract stability without freezing exact counts.",
    ),
    Step6StressCase(
        label="slow_mean_reversion",
        params=HestonParams(kappa=0.35, vbar=0.04, eta=0.55, rho=-0.70, v=0.05),
        tau=0.75,
        review_note="Slow mean reversion seed kept for explicit owner review.",
    ),
    Step6StressCase(
        label="small_v0",
        params=HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.005),
        tau=0.75,
        review_note="Small initial variance seed kept for explicit owner review.",
    ),
)

FROZEN_ACCEPTANCE_SLICES: tuple[FrozenAcceptanceSlice, ...] = (
    FrozenAcceptanceSlice(
        label="ordinary",
        case_label="long_maturity",
        review_note=(
            "Ordinary frozen acceptance slice: should remain mostly clean under "
            "the fixed review harness."
        ),
        max_backend_discrepancy=5.0e-5,
        max_config_price_span=5.0e-5,
        max_suspicious_strikes=0,
        max_non_ok_severity_count=0,
        max_parameter_sensitivity_strikes=0,
        require_any_review_signal=False,
    ),
    FrozenAcceptanceSlice(
        label="stress",
        case_label="high_vol_of_vol",
        review_note=(
            "Stress frozen acceptance slice: warnings are allowed, but the "
            "finite outputs must remain stable under backend and config review."
        ),
        max_backend_discrepancy=5.0e-4,
        max_config_price_span=5.0e-4,
        max_suspicious_strikes=4,
        max_non_ok_severity_count=3,
        max_parameter_sensitivity_strikes=2,
        require_any_review_signal=True,
    ),
)


def provisional_stress_case(label: str) -> Step6StressCase:
    for case in PROVISIONAL_STRESS_CASES:
        if case.label == label:
            return case
    raise KeyError(f"Unknown Step 6 stress case: {label!r}.")


def frozen_acceptance_slice(label: str) -> FrozenAcceptanceSlice:
    for case in FROZEN_ACCEPTANCE_SLICES:
        if case.label == label:
            return case
    raise KeyError(f"Unknown frozen acceptance slice: {label!r}.")


def build_frozen_acceptance_report(
    acceptance: FrozenAcceptanceSlice,
    *,
    primary_backend: HestonDiagnosticsBackend = "gauss_legendre",
    comparison_backend: HestonDiagnosticsBackend = "quad",
) -> HestonDiagnosticsReport:
    case = provisional_stress_case(acceptance.case_label)
    return run_heston_pricing_diagnostics(
        strike=PROVISIONAL_STRIKE_GRID,
        tau=case.tau,
        market=PROVISIONAL_MARKET,
        params=case.params,
        kind=OptionType.CALL,
        backend=primary_backend,
        comparison_backend=comparison_backend,
        use_recommended_cfg=acceptance.use_recommended_cfg,
        parameter_perturbations=provisional_parameter_perturbations(case.params),
    )


def provisional_config_sweep_cases(
    *,
    primary_backend: HestonDiagnosticsBackend = "gauss_legendre",
    comparison_backend: HestonDiagnosticsBackend = "quad",
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []

    primary_case: dict[str, object] = {
        "label": "primary",
        "backend": primary_backend,
    }
    if primary_backend == "gauss_legendre":
        primary_case["quad_cfg"] = PROVISIONAL_GAUSS_LEGENDRE_CFG
    cases.append(primary_case)

    comparison_case: dict[str, object] = {
        "label": "comparison",
        "backend": comparison_backend,
    }
    if comparison_backend == "gauss_legendre":
        comparison_case["quad_cfg"] = PROVISIONAL_GAUSS_LEGENDRE_CFG
    cases.append(comparison_case)

    return cases


def provisional_parameter_perturbations(
    params: HestonParams,
) -> list[dict[str, object]]:
    return [
        {
            "label": "eta_up",
            "parameter": "eta",
            "direction": "up",
            "bump": 0.01,
            "params": HestonParams(
                kappa=params.kappa,
                vbar=params.vbar,
                eta=params.eta + 0.01,
                rho=params.rho,
                v=params.v,
            ),
        },
        {
            "label": "rho_down",
            "parameter": "rho",
            "direction": "down",
            "bump": -0.01,
            "params": HestonParams(
                kappa=params.kappa,
                vbar=params.vbar,
                eta=params.eta,
                rho=max(params.rho - 0.01, -0.999),
                v=params.v,
            ),
        },
    ]


def build_pricing_report(
    case: Step6StressCase,
    *,
    primary_backend: HestonDiagnosticsBackend = "gauss_legendre",
    comparison_backend: HestonDiagnosticsBackend = "quad",
) -> HestonDiagnosticsReport:
    quad_cfg = (
        PROVISIONAL_GAUSS_LEGENDRE_CFG if primary_backend == "gauss_legendre" else None
    )
    comparison_quad_cfg = (
        PROVISIONAL_GAUSS_LEGENDRE_CFG
        if comparison_backend == "gauss_legendre"
        else None
    )

    return run_heston_pricing_diagnostics(
        strike=PROVISIONAL_STRIKE_GRID,
        tau=case.tau,
        market=PROVISIONAL_MARKET,
        params=case.params,
        kind=OptionType.CALL,
        backend=primary_backend,
        quad_cfg=quad_cfg,
        comparison_backend=comparison_backend,
        comparison_quad_cfg=comparison_quad_cfg,
        config_sweep_cases=provisional_config_sweep_cases(
            primary_backend=primary_backend,
            comparison_backend=comparison_backend,
        ),
        parameter_perturbations=provisional_parameter_perturbations(case.params),
    )
