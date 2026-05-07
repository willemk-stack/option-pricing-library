"""Notebook-facing Step 3 strike-slice diagnostics built on Step 2 outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...market.curves import PricingContext
from ...models.heston import HestonParams, recommend_heston_quadrature_config
from ...models.heston.fourier import (
    QuadratureQuality,
    _default_heston_quadrature_config,
)
from ...numerics.quadrature import CompositeRule, QuadratureConfig
from ...pricers.heston import (
    heston_price_call_from_ctx,
    heston_price_put_from_ctx,
)
from ...types import MarketData, OptionSpec, OptionType
from ...vol.implied_vol_scalar import implied_vol_bs
from ._provenance import backend_config_meta
from .contracts import (
    HESTON_SLICE_TABLE_COLUMNS,
    HESTON_SUMMARY_METRICS,
    HestonDiagnosticsBackend,
)
from .integration import (
    integration_config_sweep,
    integration_diagnostics,
    probability_slice_with_diagnostics,
)
from .models import (
    HestonBackendComparisonDiagnostics,
    HestonDiagnosticsReport,
    HestonIntegrationDiagnosticsBundle,
    HestonPriceSliceDiagnostics,
)
from .report import run_heston_slice_diagnostics

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _BackendSliceResult:
    backend: HestonDiagnosticsBackend
    quad_cfg: QuadratureConfig | None
    rule: CompositeRule | None
    p0: HestonIntegrationDiagnosticsBundle
    p1: HestonIntegrationDiagnosticsBundle
    price: FloatArray
    implied_vol: FloatArray


@dataclass(frozen=True, slots=True)
class _ResolvedBackendConfig:
    backend: HestonDiagnosticsBackend
    quad_cfg: QuadratureConfig | None
    rule: CompositeRule | None
    config_resolution: str | None


@dataclass(frozen=True, slots=True)
class _AcceptanceStrikePolicy:
    """Release acceptance Step 6 thresholds.

    These values stay centralized so suspiciousness flags, ranked summaries,
    and the frozen acceptance slices all use the same release acceptance
    semantics.
    """

    backend_price_diff_abs: float = 1.0e-4
    smoothness_linear_residual: float = 0.08
    discontinuity_local_jump: float = 0.12
    config_price_span_abs: float = 1.0e-4
    perturbation_relative_price_change: float = 0.10
    perturbation_absolute_price_change: float = 5.0e-4
    positive_parameter_relative_bump: float = 0.01
    positive_parameter_absolute_floor: float = 1.0e-4
    rho_absolute_bump: float = 0.01


_DEFAULT_ACCEPTANCE_POLICY = _AcceptanceStrikePolicy()


def _coerce_columns(columns: Mapping[str, Any], *, label: str) -> pd.DataFrame:
    if not columns:
        raise ValueError(f"{label} must contain at least one column.")
    return pd.DataFrame(
        {
            str(name): np.atleast_1d(np.asarray(values))
            for name, values in columns.items()
        }
    )


def _severity_rank(value: str) -> int:
    order = {"ok": 0, "info": 1, "warning": 2, "severe": 3, "critical": 4}
    return order.get(str(value), -1)


def _worst_severity(values: Sequence[str]) -> str:
    if not values:
        return "ok"
    return max(values, key=_severity_rank)


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _normalize_strike_slice(
    strike: float | Sequence[float] | np.ndarray,
) -> FloatArray:
    strike_arr = np.asarray(strike, dtype=np.float64).reshape(-1)
    if strike_arr.size == 0:
        raise ValueError("strike must contain at least one value.")
    if not np.all(np.isfinite(strike_arr)):
        raise ValueError("strike values must be finite.")
    if np.any(strike_arr <= 0.0):
        raise ValueError("strike values must be positive.")
    return np.asarray(np.sort(strike_arr), dtype=np.float64)


def _coerce_backend(value: object) -> HestonDiagnosticsBackend:
    if value == "gauss_legendre":
        return "gauss_legendre"
    if value == "quad":
        return "quad"
    raise ValueError(
        "backend must be one of 'gauss_legendre' or 'quad'. " f"Got {value!r}."
    )


def _resolve_backend_config(
    *,
    backend: HestonDiagnosticsBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
    x: float,
    tau: float,
    params: HestonParams,
    recommended_quality: QuadratureQuality | None = None,
) -> _ResolvedBackendConfig:
    resolved_backend = _coerce_backend(backend)

    if resolved_backend != "gauss_legendre":
        return _ResolvedBackendConfig(
            backend=resolved_backend,
            quad_cfg=quad_cfg,
            rule=rule,
            config_resolution=None,
        )

    if quad_cfg is not None and rule is not None:
        raise ValueError(
            "Pass either quad_cfg or rule, not both. "
            "If you already built a rule, it is the authoritative discretization."
        )

    if quad_cfg is not None:
        return _ResolvedBackendConfig(
            backend=resolved_backend,
            quad_cfg=quad_cfg,
            rule=None,
            config_resolution="explicit_quad_cfg",
        )

    if rule is not None:
        return _ResolvedBackendConfig(
            backend=resolved_backend,
            quad_cfg=None,
            rule=rule,
            config_resolution="explicit_rule",
        )

    if recommended_quality is not None:
        recommended_cfg = recommend_heston_quadrature_config(
            x=x,
            tau=tau,
            params=params,
            quality=recommended_quality,
        )
        return _ResolvedBackendConfig(
            backend=resolved_backend,
            quad_cfg=recommended_cfg,
            rule=None,
            config_resolution=f"recommended_{recommended_quality}",
        )

    return _ResolvedBackendConfig(
        backend=resolved_backend,
        quad_cfg=_default_heston_quadrature_config(),
        rule=None,
        config_resolution="default_hard_coded",
    )


def _price_from_probabilities(
    *,
    kind: OptionType,
    strike: FloatArray,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    backend: HestonDiagnosticsBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
    p0: FloatArray,
    p1: FloatArray,
) -> FloatArray:
    if kind == OptionType.CALL:
        prices = heston_price_call_from_ctx(
            strike=strike,
            ctx=ctx,
            tau=tau,
            params=params,
            p0=p0,
            p1=p1,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )
        return np.asarray(prices, dtype=np.float64)

    if kind == OptionType.PUT:
        prices = heston_price_put_from_ctx(
            strike=strike,
            tau=tau,
            ctx=ctx,
            params=params,
            p0=p0,
            p1=p1,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )
        return np.asarray(prices, dtype=np.float64)

    raise ValueError(f"Unsupported option kind: {kind!r}")


def _implied_vol_array(
    *,
    prices: np.ndarray,
    strikes: np.ndarray,
    tau: float,
    kind: OptionType,
    market: MarketData | PricingContext,
) -> np.ndarray:
    implied_vol = np.full(strikes.shape, np.nan, dtype=np.float64)
    for idx, (strike_i, price_i) in enumerate(zip(strikes, prices, strict=True)):
        spec = OptionSpec(kind=kind, strike=float(strike_i), expiry=float(tau))
        try:
            implied_vol[idx] = float(
                implied_vol_bs(
                    mkt_price=float(price_i),
                    spec=spec,
                    market=market,
                )
            )
        except Exception:  # pragma: no cover - defensive diagnostics fallback
            implied_vol[idx] = np.nan
    return implied_vol


def _numeric_column(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        return np.full(len(table), np.nan, dtype=np.float64)
    return np.asarray(pd.to_numeric(table[column], errors="coerce"), dtype=np.float64)


def _string_column(
    table: pd.DataFrame, column: str, *, default: str = ""
) -> np.ndarray:
    if column not in table.columns:
        return np.full(len(table), default, dtype=object)
    return np.asarray(table[column].fillna(default).astype(str), dtype=object)


def _combine_label_columns(left: np.ndarray, right: np.ndarray) -> list[str]:
    combined: list[str] = []
    for lhs, rhs in zip(left, right, strict=True):
        labels = [label for label in (str(lhs), str(rhs)) if label and label != "none"]
        if not labels:
            combined.append("none")
        else:
            seen: list[str] = []
            for label in labels:
                if label not in seen:
                    seen.append(label)
            combined.append("; ".join(seen))
    return combined


def _rowwise_nanmax(values: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(values)
    result = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    for idx in range(stacked.shape[1]):
        finite = stacked[:, idx][np.isfinite(stacked[:, idx])]
        if finite.size:
            result[idx] = float(np.max(finite))
    return result


def _nanmax_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _perturbation_instability_mask(
    *,
    perturbation_abs_price_change: np.ndarray,
    perturbation_rel_price_change: np.ndarray,
    policy: _AcceptanceStrikePolicy,
) -> np.ndarray:
    return np.asarray(
        (perturbation_rel_price_change > policy.perturbation_relative_price_change)
        & (perturbation_abs_price_change > policy.perturbation_absolute_price_change),
        dtype=np.bool_,
    )


def _continuity_signals(
    *,
    strike: np.ndarray,
    implied_vol: np.ndarray,
    policy: _AcceptanceStrikePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smoothness = np.full(strike.shape, np.nan, dtype=np.float64)
    discontinuity = np.full(strike.shape, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(implied_vol)
    if np.count_nonzero(finite_mask) >= 2:
        for idx in range(len(strike)):
            diffs: list[float] = []
            if idx > 0 and finite_mask[idx] and finite_mask[idx - 1]:
                scale = max(
                    abs(float(implied_vol[idx])),
                    abs(float(implied_vol[idx - 1])),
                    1.0e-8,
                )
                diffs.append(
                    abs(float(implied_vol[idx] - implied_vol[idx - 1])) / scale
                )
            if idx + 1 < len(strike) and finite_mask[idx] and finite_mask[idx + 1]:
                scale = max(
                    abs(float(implied_vol[idx])),
                    abs(float(implied_vol[idx + 1])),
                    1.0e-8,
                )
                diffs.append(
                    abs(float(implied_vol[idx + 1] - implied_vol[idx])) / scale
                )
            if diffs:
                discontinuity[idx] = float(max(diffs))

    if np.count_nonzero(finite_mask) >= 3:
        for idx in range(1, len(strike) - 1):
            if not (finite_mask[idx - 1] and finite_mask[idx] and finite_mask[idx + 1]):
                continue
            span = float(strike[idx + 1] - strike[idx - 1])
            if span <= 0.0:
                continue
            weight = float((strike[idx] - strike[idx - 1]) / span)
            predicted = float(implied_vol[idx - 1]) + weight * float(
                implied_vol[idx + 1] - implied_vol[idx - 1]
            )
            scale = max(abs(float(implied_vol[idx])), 1.0e-8)
            smoothness[idx] = abs(float(implied_vol[idx]) - predicted) / scale

    smoothness_flag = np.asarray(
        np.isfinite(smoothness) & (smoothness > policy.smoothness_linear_residual),
        dtype=np.bool_,
    )
    discontinuity_flag = np.asarray(
        np.isfinite(discontinuity) & (discontinuity > policy.discontinuity_local_jump),
        dtype=np.bool_,
    )
    return smoothness, discontinuity, smoothness_flag, discontinuity_flag


def _append_case(
    cases: list[dict[str, Any]],
    *,
    label: str,
    backend: HestonDiagnosticsBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
    config_resolution: str | None = None,
) -> None:
    for case in cases:
        if case["label"] == label:
            return
    case = {
        "label": label,
        "backend": backend,
        "quad_cfg": quad_cfg,
        "rule": rule,
    }
    if config_resolution is not None:
        case["config_resolution"] = config_resolution
    cases.append(case)


def _config_sweep_case_details(case: Mapping[str, Any]) -> dict[str, Any]:
    backend = _coerce_backend(case.get("backend", "gauss_legendre"))
    quad_cfg = case.get("quad_cfg")
    rule = case.get("rule")
    return backend_config_meta(
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        config_resolution=case.get("config_resolution"),
    )


def _default_config_sweep_cases(
    *,
    x: FloatArray,
    tau: float,
    params: HestonParams,
    primary_config: _ResolvedBackendConfig,
    comparison_config: _ResolvedBackendConfig,
) -> list[dict[str, Any]]:
    max_abs_x = float(np.max(np.abs(x)))
    balanced_cfg = recommend_heston_quadrature_config(
        x=max_abs_x,
        tau=tau,
        params=params,
        quality="balanced",
    )
    robust_cfg = recommend_heston_quadrature_config(
        x=max_abs_x,
        tau=tau,
        params=params,
        quality="robust",
    )

    cases: list[dict[str, Any]] = []
    _append_case(
        cases,
        label="primary",
        backend=primary_config.backend,
        quad_cfg=primary_config.quad_cfg,
        rule=primary_config.rule,
        config_resolution=primary_config.config_resolution,
    )
    _append_case(
        cases,
        label="comparison",
        backend=comparison_config.backend,
        quad_cfg=comparison_config.quad_cfg,
        rule=comparison_config.rule,
        config_resolution=comparison_config.config_resolution,
    )
    _append_case(
        cases,
        label="gauss_robust",
        backend="gauss_legendre",
        quad_cfg=robust_cfg,
        rule=None,
        config_resolution="recommended_robust",
    )
    _append_case(
        cases,
        label="gauss_balanced",
        backend="gauss_legendre",
        quad_cfg=balanced_cfg,
        rule=None,
        config_resolution="recommended_balanced",
    )
    return cases


def _parameter_perturbation_cases(
    *,
    params: HestonParams,
    policy: _AcceptanceStrikePolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name in ("kappa", "vbar", "eta", "v"):
        value = float(getattr(params, name))
        bump = max(
            abs(value) * policy.positive_parameter_relative_bump,
            policy.positive_parameter_absolute_floor,
        )
        for direction, sign in (("down", -1.0), ("up", 1.0)):
            candidate = dict(
                kappa=float(params.kappa),
                vbar=float(params.vbar),
                eta=float(params.eta),
                rho=float(params.rho),
                v=float(params.v),
            )
            candidate[name] = max(value + sign * bump, 1.0e-12)
            rows.append(
                {
                    "label": f"{name}_{direction}",
                    "parameter": name,
                    "direction": direction,
                    "bump": float(sign * bump),
                    "params": HestonParams(**candidate),
                }
            )

    for direction, sign in (("down", -1.0), ("up", 1.0)):
        rho = float(
            np.clip(
                float(params.rho) + sign * policy.rho_absolute_bump,
                -0.999999,
                0.999999,
            )
        )
        rows.append(
            {
                "label": f"rho_{direction}",
                "parameter": "rho",
                "direction": direction,
                "bump": float(sign * policy.rho_absolute_bump),
                "params": HestonParams(
                    kappa=float(params.kappa),
                    vbar=float(params.vbar),
                    eta=float(params.eta),
                    rho=rho,
                    v=float(params.v),
                ),
            }
        )

    return rows


def _compute_backend_slice(
    *,
    strike: FloatArray,
    x: FloatArray,
    tau: float,
    params: HestonParams,
    ctx: PricingContext,
    market: MarketData | PricingContext,
    kind: OptionType,
    backend: HestonDiagnosticsBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> _BackendSliceResult:
    p0 = integration_diagnostics(
        x=x,
        tau=tau,
        params=params,
        j=0,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        strike=strike,
    )
    p1 = integration_diagnostics(
        x=x,
        tau=tau,
        params=params,
        j=1,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        strike=strike,
    )

    p0_values = p0.probability.table["probability"].to_numpy(dtype=np.float64)
    p1_values = p1.probability.table["probability"].to_numpy(dtype=np.float64)
    price = _price_from_probabilities(
        kind=kind,
        strike=strike,
        ctx=ctx,
        tau=tau,
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        p0=p0_values,
        p1=p1_values,
    )
    implied_vol = _implied_vol_array(
        prices=price,
        strikes=strike,
        tau=tau,
        kind=kind,
        market=market,
    )
    return _BackendSliceResult(
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        p0=p0,
        p1=p1,
        price=price,
        implied_vol=implied_vol,
    )


def price_slice_with_diagnostics(
    *,
    slice_table: pd.DataFrame | None = None,
    slice_columns: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonPriceSliceDiagnostics:
    """Freeze the canonical notebook-facing strike-slice contract.

    Use this when downstream code already has the slice table and needs the
    stable ``HestonPriceSliceDiagnostics`` wrapper without rerunning pricing.
    The required Step 1 columns must be present; any additional review columns
    are preserved after that canonical prefix.

    This function packages existing diagnostics only. It does not calculate
    prices, warnings, or continuity signals by itself.
    """

    if slice_table is None and slice_columns is None:
        raise ValueError("Provide slice_table or slice_columns.")
    if slice_table is not None and slice_columns is not None:
        raise ValueError("Pass either slice_table or slice_columns, not both.")

    if slice_table is not None:
        table = slice_table.copy()
    else:
        assert slice_columns is not None
        table = _coerce_columns(slice_columns, label="slice_columns")

    missing = [name for name in HESTON_SLICE_TABLE_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(
            "slice diagnostics must expose the canonical slice columns: "
            f"{', '.join(missing)}."
        )

    return HestonPriceSliceDiagnostics(
        meta={} if meta is None else dict(meta),
        table=table,
        arrays={} if arrays is None else dict(arrays),
    )


def compare_backend_slice(
    *,
    comparison_table: pd.DataFrame | None = None,
    comparison_columns: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonBackendComparisonDiagnostics:
    """Package backend-comparison data for a strike slice.

    The helper keeps the comparison table and aligned arrays in a lightweight
    runtime artifact so notebooks can inspect backend disagreement without
    re-running the comparison logic.
    """

    if comparison_table is None and comparison_columns is None:
        raise ValueError("Provide comparison_table or comparison_columns.")
    if comparison_table is not None and comparison_columns is not None:
        raise ValueError(
            "Pass either comparison_table or comparison_columns, not both."
        )

    if comparison_table is not None:
        table = comparison_table.copy()
    else:
        assert comparison_columns is not None
        table = _coerce_columns(
            comparison_columns,
            label="comparison_columns",
        )

    return HestonBackendComparisonDiagnostics(
        meta={} if meta is None else dict(meta),
        table=table,
        arrays={} if arrays is None else dict(arrays),
    )


def run_heston_pricing_diagnostics(
    *,
    strike: float | Sequence[float] | np.ndarray,
    tau: float,
    market: MarketData | PricingContext,
    params: HestonParams,
    kind: OptionType = OptionType.CALL,
    backend: HestonDiagnosticsBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
    comparison_backend: HestonDiagnosticsBackend | None = None,
    comparison_quad_cfg: QuadratureConfig | None = None,
    comparison_rule: CompositeRule | None = None,
    use_recommended_cfg: bool = False,
    config_sweep_cases: Sequence[Mapping[str, Any]] | None = None,
    parameter_perturbations: Sequence[Mapping[str, Any]] | None = None,
    top_n: int = 10,
) -> HestonDiagnosticsReport:
    """Run the notebook-facing Heston strike-slice diagnostics workflow.

    This is the one-call entrypoint used by review notebooks. It prices a
    representative strike slice with one primary backend, runs the comparison
    backend, collects the packaged ``P0``/``P1`` diagnostics, builds the
    convergence/config-sweep summaries, and returns a
    :class:`~option_pricing.diagnostics.heston.models.HestonDiagnosticsReport`
    with the frozen ``meta`` / ``tables`` / ``arrays`` topology.

    The returned report is designed for stability review:

    - convergence and config-sweep behavior
    - smoothness and continuity signals
    - backend differences
    - visible failure modes through worst-strike and worst-panel tables

    It is intentionally conservative about claims. The report does not prove
    prices are correct, does not validate economic smile realism, and does not
    cover calibration, optimizer, or Monte Carlo diagnostics. Thresholds used
    for suspiciousness and continuity flags are part of the release acceptance
    policy and are surfaced in ``report.meta["acceptance_policy"]``.
    ``report.meta["provisional_policy"]`` is retained as a compatibility alias.
    """

    if tau <= 0.0:
        raise ValueError("tau must be > 0 for pricing diagnostics.")

    policy = _DEFAULT_ACCEPTANCE_POLICY
    ctx = _to_ctx(market)
    strike_arr = _normalize_strike_slice(strike)
    forward = float(ctx.fwd(tau))
    x = np.asarray(np.log(forward / strike_arr), dtype=np.float64)
    max_abs_x = float(np.max(np.abs(x)))

    resolved_comparison_backend: HestonDiagnosticsBackend = (
        "quad" if backend == "gauss_legendre" else "gauss_legendre"
    )
    if comparison_backend is not None:
        resolved_comparison_backend = comparison_backend

    primary_config = _resolve_backend_config(
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        x=max_abs_x,
        tau=tau,
        params=params,
        recommended_quality="balanced" if use_recommended_cfg else None,
    )
    comparison_config = _resolve_backend_config(
        backend=resolved_comparison_backend,
        quad_cfg=comparison_quad_cfg,
        rule=comparison_rule,
        x=max_abs_x,
        tau=tau,
        params=params,
        recommended_quality=(
            "robust"
            if use_recommended_cfg and resolved_comparison_backend == "gauss_legendre"
            else None
        ),
    )
    primary_config_meta = backend_config_meta(
        backend=primary_config.backend,
        quad_cfg=primary_config.quad_cfg,
        rule=primary_config.rule,
        config_resolution=primary_config.config_resolution,
    )
    comparison_config_meta = backend_config_meta(
        backend=comparison_config.backend,
        quad_cfg=comparison_config.quad_cfg,
        rule=comparison_config.rule,
        config_resolution=comparison_config.config_resolution,
    )

    primary = _compute_backend_slice(
        strike=strike_arr,
        x=x,
        tau=tau,
        params=params,
        ctx=ctx,
        market=market,
        kind=kind,
        backend=primary_config.backend,
        quad_cfg=primary_config.quad_cfg,
        rule=primary_config.rule,
    )
    comparison = _compute_backend_slice(
        strike=strike_arr,
        x=x,
        tau=tau,
        params=params,
        ctx=ctx,
        market=market,
        kind=kind,
        backend=comparison_config.backend,
        quad_cfg=comparison_config.quad_cfg,
        rule=comparison_config.rule,
    )

    if config_sweep_cases is None:
        config_sweep_cases = _default_config_sweep_cases(
            x=x,
            tau=tau,
            params=params,
            primary_config=primary_config,
            comparison_config=comparison_config,
        )

    config_sweep_p0, p0_cases = integration_config_sweep(
        x=x,
        tau=tau,
        params=params,
        j=0,
        cases=config_sweep_cases,
        strike=strike_arr,
    )
    config_sweep_p1, p1_cases = integration_config_sweep(
        x=x,
        tau=tau,
        params=params,
        j=1,
        cases=config_sweep_cases,
        strike=strike_arr,
    )

    config_price_by_label: dict[str, FloatArray] = {}
    config_rows: list[dict[str, Any]] = []
    baseline_price: np.ndarray | None = None
    for case in config_sweep_cases:
        label = str(case.get("label", "case"))
        p0_case = p0_cases[label]
        p1_case = p1_cases[label]
        case_backend = _coerce_backend(case.get("backend", backend))
        case_quad_cfg = case.get("quad_cfg")
        case_rule = case.get("rule")
        case_price = _price_from_probabilities(
            kind=kind,
            strike=strike_arr,
            ctx=ctx,
            tau=tau,
            params=params,
            backend=case_backend,
            quad_cfg=case_quad_cfg,
            rule=case_rule,
            p0=p0_case.probability.table["probability"].to_numpy(dtype=np.float64),
            p1=p1_case.probability.table["probability"].to_numpy(dtype=np.float64),
        )
        config_price_by_label[label] = case_price
        if baseline_price is None:
            baseline_price = case_price.copy()
        abs_diff = np.abs(case_price - baseline_price)
        p0_row = config_sweep_p0.loc[config_sweep_p0["config_label"] == label].iloc[0]
        p1_row = config_sweep_p1.loc[config_sweep_p1["config_label"] == label].iloc[0]
        case_details = _config_sweep_case_details(case)
        config_rows.append(
            {
                "config_label": label,
                "backend": case_backend,
                "config_resolution": case_details["config_resolution"],
                "resolved_u_max": case_details["u_max"],
                "resolved_n_panels": case_details["n_panels"],
                "resolved_nodes_per_panel": case_details["nodes_per_panel"],
                "resolved_panel_spacing": case_details["panel_spacing"],
                "resolved_cluster_strength": case_details["cluster_strength"],
                "p0_max_severity": p0_row["max_severity"],
                "p1_max_severity": p1_row["max_severity"],
                "p0_warning_point_count": int(p0_row["warning_point_count"]),
                "p1_warning_point_count": int(p1_row["warning_point_count"]),
                "max_abs_price_diff_vs_baseline": float(np.nanmax(abs_diff)),
                "mean_abs_price_diff_vs_baseline": float(np.nanmean(abs_diff)),
                "max_abs_probability_diff_p0": float(
                    p0_row["max_abs_probability_diff_vs_baseline"]
                ),
                "max_abs_probability_diff_p1": float(
                    p1_row["max_abs_probability_diff_vs_baseline"]
                ),
                "notes": "",
            }
        )

    config_sweep_table = pd.DataFrame(config_rows)
    config_price_span = np.full(strike_arr.shape, np.nan, dtype=np.float64)
    if config_price_by_label:
        price_stack = np.vstack(list(config_price_by_label.values()))
        baseline_price = price_stack[0, :]
        config_price_span = np.max(
            np.abs(price_stack - baseline_price[None, :]), axis=0
        )

    perturbation_cases = (
        _parameter_perturbation_cases(params=params, policy=policy)
        if parameter_perturbations is None
        else [dict(case) for case in parameter_perturbations]
    )
    perturbation_rows: list[dict[str, Any]] = []
    perturbation_price_by_label: dict[str, FloatArray] = {}
    perturbation_abs_price_change = np.zeros(strike_arr.shape, dtype=np.float64)
    perturbation_rel_price_change = np.zeros(strike_arr.shape, dtype=np.float64)
    baseline_primary_price = primary.price.copy()
    perturbation_config_meta = backend_config_meta(
        backend=primary_config.backend,
        quad_cfg=primary_config.quad_cfg,
        rule=primary_config.rule,
        config_resolution=primary_config.config_resolution,
    )

    for case in perturbation_cases:
        label = str(case["label"])
        perturbed_params = case["params"]
        perturbed = _compute_backend_slice(
            strike=strike_arr,
            x=x,
            tau=tau,
            params=perturbed_params,
            ctx=ctx,
            market=market,
            kind=kind,
            backend=primary_config.backend,
            quad_cfg=primary_config.quad_cfg,
            rule=primary_config.rule,
        )
        abs_diff = np.abs(perturbed.price - baseline_primary_price)
        rel_diff = abs_diff / np.maximum(np.abs(baseline_primary_price), 1.0e-8)
        perturbation_price_by_label[label] = perturbed.price
        perturbation_abs_price_change = np.maximum(
            perturbation_abs_price_change, abs_diff
        )
        perturbation_rel_price_change = np.maximum(
            perturbation_rel_price_change, rel_diff
        )
        perturbation_rows.append(
            {
                "label": label,
                "parameter": case["parameter"],
                "direction": case["direction"],
                "bump": float(case["bump"]),
                "backend": perturbation_config_meta["backend"],
                "config_resolution": perturbation_config_meta["config_resolution"],
                "resolved_u_max": perturbation_config_meta["u_max"],
                "resolved_n_panels": perturbation_config_meta["n_panels"],
                "resolved_nodes_per_panel": perturbation_config_meta["nodes_per_panel"],
                "resolved_panel_spacing": perturbation_config_meta["panel_spacing"],
                "resolved_cluster_strength": perturbation_config_meta[
                    "cluster_strength"
                ],
                "max_abs_price_change": float(np.nanmax(abs_diff)),
                "max_relative_price_change": float(np.nanmax(rel_diff)),
                "notes": (
                    "Release acceptance policy: perturbation magnitudes use "
                    "frozen acceptance thresholds."
                ),
            }
        )

    perturbation_table = pd.DataFrame(perturbation_rows)

    smoothness_signal, discontinuity_signal, smoothness_flag, discontinuity_flag = (
        _continuity_signals(
            strike=strike_arr,
            implied_vol=primary.implied_vol,
            policy=policy,
        )
    )

    p0_table = primary.p0.probability.table.reset_index(drop=True)
    p1_table = primary.p1.probability.table.reset_index(drop=True)
    comparison_p0_table = comparison.p0.probability.table.reset_index(drop=True)
    comparison_p1_table = comparison.p1.probability.table.reset_index(drop=True)

    warning_count_p0 = p0_table["warning_count"].to_numpy(dtype=np.int64)
    warning_count_p1 = p1_table["warning_count"].to_numpy(dtype=np.int64)
    severity_p0 = _string_column(p0_table, "severity", default="ok")
    severity_p1 = _string_column(p1_table, "severity", default="ok")
    warning_labels_p0 = _string_column(p0_table, "warning_labels", default="none")
    warning_labels_p1 = _string_column(p1_table, "warning_labels", default="none")
    tail_fraction_p0 = _numeric_column(p0_table, "tail_fraction")
    tail_fraction_p1 = _numeric_column(p1_table, "tail_fraction")
    cancellation_p0 = _numeric_column(p0_table, "cancellation_ratio")
    cancellation_p1 = _numeric_column(p1_table, "cancellation_ratio")

    severity = np.asarray(
        [
            _worst_severity([str(lhs), str(rhs)])
            for lhs, rhs in zip(severity_p0, severity_p1, strict=True)
        ],
        dtype=object,
    )
    severity_rank = np.asarray(
        [_severity_rank(str(value)) for value in severity],
        dtype=np.int64,
    )
    backend_diff = np.asarray(
        np.abs(primary.price - comparison.price), dtype=np.float64
    )
    warning_count = warning_count_p0 + warning_count_p1
    tail_fraction = _rowwise_nanmax([tail_fraction_p0, tail_fraction_p1])
    cancellation_ratio = _rowwise_nanmax([cancellation_p0, cancellation_p1])
    parameter_sensitivity_flag = _perturbation_instability_mask(
        perturbation_abs_price_change=perturbation_abs_price_change,
        perturbation_rel_price_change=perturbation_rel_price_change,
        policy=policy,
    )

    suspicious_flag = np.asarray(
        (backend_diff > policy.backend_price_diff_abs)
        | (severity_rank >= _severity_rank("warning"))
        | smoothness_flag
        | discontinuity_flag
        | (config_price_span > policy.config_price_span_abs),
        dtype=np.bool_,
    )
    suspicious_reasons: list[str] = []
    parameter_sensitivity_reasons: list[str] = []
    for idx in range(len(strike_arr)):
        reasons: list[str] = []
        if backend_diff[idx] > policy.backend_price_diff_abs:
            reasons.append("backend price disagreement")
        if severity_rank[idx] >= _severity_rank("warning"):
            reasons.append("integration warning severity >= warning")
        if smoothness_flag[idx]:
            reasons.append("smoothness flag")
        if discontinuity_flag[idx]:
            reasons.append("discontinuity flag")
        if config_price_span[idx] > policy.config_price_span_abs:
            reasons.append("config sweep price span")
        suspicious_reasons.append("; ".join(reasons) if reasons else "none")
        parameter_sensitivity_reasons.append(
            "parameter perturbation instability"
            if parameter_sensitivity_flag[idx]
            else "none"
        )

    slice_table = pd.DataFrame(
        {
            "strike": strike_arr,
            "log_moneyness": x,
            "price": primary.price,
            "implied_vol": primary.implied_vol,
            "warning_count": warning_count,
            "severity": severity,
            "tail_fraction": tail_fraction,
            "cancellation_ratio": cancellation_ratio,
            "backend_diff": backend_diff,
            "smoothness_flag": smoothness_flag,
            "discontinuity_flag": discontinuity_flag,
            "primary_backend": np.full(len(strike_arr), backend, dtype=object),
            "comparison_backend": np.full(
                len(strike_arr),
                resolved_comparison_backend,
                dtype=object,
            ),
            "warning_count_p0": warning_count_p0,
            "warning_count_p1": warning_count_p1,
            "severity_p0": severity_p0,
            "severity_p1": severity_p1,
            "warning_labels_p0": warning_labels_p0,
            "warning_labels_p1": warning_labels_p1,
            "combined_warning_labels": _combine_label_columns(
                warning_labels_p0,
                warning_labels_p1,
            ),
            "tail_fraction_p0": tail_fraction_p0,
            "tail_fraction_p1": tail_fraction_p1,
            "cancellation_ratio_p0": cancellation_p0,
            "cancellation_ratio_p1": cancellation_p1,
            "price_comparison_backend": comparison.price,
            "implied_vol_comparison_backend": comparison.implied_vol,
            "backend_price_diff_signed": primary.price - comparison.price,
            "config_price_span": config_price_span,
            "perturbation_max_absolute_price_change": perturbation_abs_price_change,
            "smoothness_signal": smoothness_signal,
            "discontinuity_signal": discontinuity_signal,
            "perturbation_max_relative_price_change": perturbation_rel_price_change,
            "parameter_sensitivity_flag": parameter_sensitivity_flag,
            "parameter_sensitivity_reasons": parameter_sensitivity_reasons,
            "suspicious_flag": suspicious_flag,
            "suspicious_reasons": suspicious_reasons,
        }
    )

    backend_compare_table = pd.DataFrame(
        {
            "strike": strike_arr,
            "log_moneyness": x,
            "backend_a": np.full(len(strike_arr), backend, dtype=object),
            "backend_b": np.full(
                len(strike_arr),
                resolved_comparison_backend,
                dtype=object,
            ),
            "price_a": primary.price,
            "price_b": comparison.price,
            "price_diff": primary.price - comparison.price,
            "abs_price_diff": backend_diff,
            "implied_vol_a": primary.implied_vol,
            "implied_vol_b": comparison.implied_vol,
            "implied_vol_diff": primary.implied_vol - comparison.implied_vol,
            "warning_count_a": warning_count,
            "warning_count_b": (
                comparison_p0_table["warning_count"].to_numpy(dtype=np.int64)
                + comparison_p1_table["warning_count"].to_numpy(dtype=np.int64)
            ),
            "severity_a": severity,
            "severity_b": np.asarray(
                [
                    _worst_severity([str(lhs), str(rhs)])
                    for lhs, rhs in zip(
                        _string_column(comparison_p0_table, "severity", default="ok"),
                        _string_column(comparison_p1_table, "severity", default="ok"),
                        strict=True,
                    )
                ],
                dtype=object,
            ),
        }
    )

    worst_strikes = slice_table.sort_values(
        [
            "suspicious_flag",
            "parameter_sensitivity_flag",
            "severity",
            "backend_diff",
            "config_price_span",
            "perturbation_max_relative_price_change",
            "perturbation_max_absolute_price_change",
            "warning_count",
            "tail_fraction",
            "cancellation_ratio",
        ],
        key=lambda col: col.map(_severity_rank) if col.name == "severity" else col,
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        ignore_index=True,
    ).head(int(top_n))
    worst_strikes = worst_strikes.copy()
    worst_strikes.insert(
        0, "rank", np.arange(1, len(worst_strikes) + 1, dtype=np.int64)
    )

    summary_lookup: dict[str, dict[str, Any]] = {}
    worst_row = slice_table.sort_values(
        [
            "suspicious_flag",
            "parameter_sensitivity_flag",
            "severity",
            "backend_diff",
            "config_price_span",
            "perturbation_max_relative_price_change",
            "perturbation_max_absolute_price_change",
            "warning_count",
        ],
        key=lambda col: col.map(_severity_rank) if col.name == "severity" else col,
        ascending=[False, False, False, False, False, False, False, False],
        ignore_index=True,
    ).head(1)
    worst_strike_value = None if worst_row.empty else float(worst_row.iloc[0]["strike"])
    worst_strike_notes = (
        "No strike diagnostics were produced."
        if worst_row.empty
        else str(
            worst_row.iloc[0]["suspicious_reasons"]
            if str(worst_row.iloc[0]["suspicious_reasons"]) != "none"
            else worst_row.iloc[0]["parameter_sensitivity_reasons"]
        )
    )
    worst_strike_severity = (
        "ok" if worst_row.empty else str(worst_row.iloc[0]["severity"])
    )
    summary_lookup["worst_strike"] = {
        "value": worst_strike_value,
        "notes": worst_strike_notes,
        "severity": worst_strike_severity,
    }
    total_warnings = int(slice_table["warning_count"].sum())
    summary_lookup["total_warnings"] = {
        "value": total_warnings,
        "notes": "Sum of P0 and P1 warning counts across the strike slice.",
        "severity": "warning" if total_warnings > 0 else "ok",
    }
    max_tail_fraction = _nanmax_or_none(
        slice_table["tail_fraction"].to_numpy(dtype=np.float64)
    )
    max_slice_severity_rank = (
        int(slice_table["severity"].map(_severity_rank).max())
        if len(slice_table)
        else 0
    )
    summary_lookup["max_tail_fraction"] = {
        "value": max_tail_fraction,
        "notes": "Max of the P0/P1 tail fractions at each strike.",
        "severity": (
            "warning" if max_slice_severity_rank >= _severity_rank("warning") else "ok"
        ),
    }
    max_cancellation_ratio = _nanmax_or_none(
        slice_table["cancellation_ratio"].to_numpy(dtype=np.float64)
    )
    summary_lookup["max_cancellation_ratio"] = {
        "value": max_cancellation_ratio,
        "notes": "Max of the P0/P1 cancellation ratios at each strike.",
        "severity": worst_strike_severity,
    }
    max_backend_discrepancy = _nanmax_or_none(backend_diff)
    summary_lookup["max_backend_discrepancy"] = {
        "value": max_backend_discrepancy,
        "notes": (
            "Absolute price difference between the primary and comparison backends."
        ),
        "severity": (
            "warning"
            if max_backend_discrepancy is not None
            and max_backend_discrepancy > policy.backend_price_diff_abs
            else "ok"
        ),
    }
    suspicious_count = int(np.count_nonzero(suspicious_flag))
    summary_lookup["suspicious_strike_count"] = {
        "value": suspicious_count,
        "notes": (
            "Release acceptance policy block used for backend diff, "
            "continuity, and config sweep thresholds across the frozen "
            "acceptance slices."
        ),
        "severity": "warning" if suspicious_count > 0 else "ok",
    }
    summary_rows = [
        {
            "metric": metric,
            "value": summary_lookup[metric]["value"],
            "notes": summary_lookup[metric]["notes"],
            "severity": summary_lookup[metric]["severity"],
        }
        for metric in HESTON_SUMMARY_METRICS
    ]
    summary_table = pd.DataFrame(summary_rows)

    price_artifact = price_slice_with_diagnostics(
        slice_table=slice_table,
        meta={
            "kind": str(kind.value),
            "tau": float(tau),
            "backend": backend,
            "comparison_backend": resolved_comparison_backend,
            "backend_config": primary_config_meta,
            "comparison_backend_config": comparison_config_meta,
            "policy": "release_acceptance_policy",
        },
        arrays={
            "primary_price": primary.price,
            "comparison_price": comparison.price,
            "primary_implied_vol": primary.implied_vol,
            "comparison_implied_vol": comparison.implied_vol,
            "smoothness_signal": smoothness_signal,
            "discontinuity_signal": discontinuity_signal,
            "perturbation_max_absolute_price_change": perturbation_abs_price_change,
            "config_price_span": config_price_span,
            "perturbation_max_relative_price_change": perturbation_rel_price_change,
            "parameter_sensitivity_flag": parameter_sensitivity_flag,
            "suspicious_flag": suspicious_flag,
        },
    )
    backend_artifact = compare_backend_slice(
        comparison_table=backend_compare_table,
        meta={
            "backend_a": backend,
            "backend_b": resolved_comparison_backend,
            "backend_a_config": primary_config_meta,
            "backend_b_config": comparison_config_meta,
            "primary_metric": "price difference",
        },
        arrays={
            "price_a": primary.price,
            "price_b": comparison.price,
            "abs_price_diff": backend_diff,
        },
    )

    return run_heston_slice_diagnostics(
        meta={
            "kind": str(kind.value),
            "tau": float(tau),
            "spot": float(ctx.spot),
            "forward": float(forward),
            "primary_backend": primary_config.backend,
            "comparison_backend": resolved_comparison_backend,
            "primary_backend_config": primary_config_meta,
            "comparison_backend_config": comparison_config_meta,
            "parameter_perturbation_backend_config": perturbation_config_meta,
            "acceptance_policy": asdict(policy),
            "provisional_policy": asdict(policy),
            "config_sweep_labels": [str(case["label"]) for case in config_sweep_cases],
            "parameter_perturbations": [
                {
                    "label": row["label"],
                    "parameter": row["parameter"],
                    "direction": row["direction"],
                    "bump": row["bump"],
                    "backend": row["backend"],
                    "config_resolution": row["config_resolution"],
                    "resolved_u_max": row["resolved_u_max"],
                    "resolved_n_panels": row["resolved_n_panels"],
                    "resolved_nodes_per_panel": row["resolved_nodes_per_panel"],
                    "resolved_panel_spacing": row["resolved_panel_spacing"],
                    "resolved_cluster_strength": row["resolved_cluster_strength"],
                }
                for row in perturbation_rows
            ],
        },
        tables={
            "summary": summary_table,
            "worst_strikes": worst_strikes,
            "config_sweep": config_sweep_table,
            "worst_panels_p0": primary.p0.tables["worst_panels"],
            "worst_panels_p1": primary.p1.tables["worst_panels"],
        },
        arrays={
            "probability_p0": dict(primary.p0.probability.arrays),
            "probability_p1": dict(primary.p1.probability.arrays),
            "comparison_probability_p0": dict(comparison.p0.probability.arrays),
            "comparison_probability_p1": dict(comparison.p1.probability.arrays),
            "config_sweep_prices": {
                label: values for label, values in config_price_by_label.items()
            },
            "parameter_perturbation_prices": {
                label: values for label, values in perturbation_price_by_label.items()
            },
            "parameter_perturbation_table": perturbation_table.to_dict(orient="list"),
        },
        price=price_artifact,
        backend_comparison=backend_artifact,
    )


__all__ = [
    "compare_backend_slice",
    "price_slice_with_diagnostics",
    "probability_slice_with_diagnostics",
    "run_heston_pricing_diagnostics",
]
