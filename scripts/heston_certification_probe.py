"""Run deterministic Heston numerical certification probes on an OPL checkout."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from option_pricing.models.heston import (
    HestonParams,
    recommend_heston_quadrature_config,
)
from option_pricing.pricers.heston import heston_price_from_ctx
from option_pricing.types import MarketData, OptionType


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _context(*, forward: float, discount_factor: float, tau: float):
    spot = float(forward)
    rate = -math.log(float(discount_factor)) / float(tau)
    return MarketData(spot=spot, rate=rate, dividend_yield=rate).to_context()


def _price(
    *,
    kind: OptionType,
    strike: float | np.ndarray,
    tau: float,
    ctx: Any,
    params: HestonParams,
    backend: str,
    quality: str | None = None,
    recommendation_x: float | None = None,
) -> float | np.ndarray:
    cfg = None
    if backend == "gauss_legendre":
        x_values = np.log(float(ctx.fwd(tau=tau)) / np.asarray(strike, dtype=float))
        max_abs_x = (
            float(np.max(np.abs(x_values)))
            if recommendation_x is None
            else abs(float(recommendation_x))
        )
        cfg = recommend_heston_quadrature_config(
            x=max_abs_x,
            tau=tau,
            params=params,
            quality=quality or "robust",
        )
    return heston_price_from_ctx(
        kind=kind,
        strike=strike,
        tau=tau,
        ctx=ctx,
        params=params,
        backend=backend,
        quad_cfg=cfg,
    )


def _tolerance(
    tolerances: dict[str, float],
    *,
    prefix: str,
    scale: float,
) -> float:
    return float(tolerances[f"{prefix}_absolute_price"]) + float(
        tolerances[f"{prefix}_relative_to_discounted_forward"]
    ) * abs(float(scale))


def _bound_tolerance(tolerances: dict[str, float], *, scale: float) -> float:
    return float(tolerances["bound_absolute_price"]) + float(
        tolerances["bound_relative_to_discounted_notional"]
    ) * abs(float(scale))


def _parity_tolerance(tolerances: dict[str, float], *, scale: float) -> float:
    return float(tolerances["parity_absolute_price"]) + float(
        tolerances["parity_relative_to_discounted_notional"]
    ) * abs(float(scale))


def _case_identity(
    *,
    params: HestonParams,
    tau: float,
    x: float,
    discount_factor: float,
    forward: float,
) -> dict[str, float | str]:
    return {
        "kappa": float(params.kappa),
        "vbar": float(params.vbar),
        "eta": float(params.eta),
        "rho": float(params.rho),
        "v0": float(params.v),
        "maturity_years": float(tau),
        "log_forward_moneyness": float(x),
        "discount_factor": float(discount_factor),
        "forward": float(forward),
    }


def run_probe(scope: dict[str, Any]) -> dict[str, Any]:
    """Execute the declared grid and return structured pass/fail evidence."""
    market = scope["grid_definition"]
    tolerances = {
        str(name): float(value) for name, value in scope["numerical_tolerances"].items()
    }
    parameter_names = ("kappa", "vbar", "eta", "rho", "v0")
    parameter_cases: list[tuple[str, tuple[float, ...]]] = []
    for regime in scope["certified_parameter_regimes"]:
        regime_values = [
            (
                float(regime["envelope"][name]["minimum"]),
                float(regime["envelope"][name]["maximum"]),
            )
            for name in parameter_names
        ]
        parameter_cases.extend(
            (str(regime["name"]), values)
            for values in itertools.product(*regime_values)
        )
    maturities = [float(value) for value in market["maturities_years"]]
    moneynesses = [float(value) for value in market["log_forward_moneyness"]]
    discount_factors = [float(value) for value in market["discount_factors"]]

    failures: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    maximums: dict[str, dict[str, Any]] = {}
    grid_case_count = 0

    def record_maximum(
        metric: str,
        value: float,
        case: dict[str, float | str],
    ) -> None:
        current = maximums.get(metric)
        if current is None or value > float(current["value"]):
            maximums[metric] = {"value": float(value), "case": case}

    def fail(
        code: str,
        case: dict[str, float | str],
        **details: object,
    ) -> None:
        if len(failures) < 100:
            failures.append({"reason_code": code, "case": case, **details})

    forward = 100.0
    for regime_name, values in parameter_cases:
        params = HestonParams(
            kappa=values[0],
            vbar=values[1],
            eta=values[2],
            rho=values[3],
            v=values[4],
        )
        for tau, x, discount_factor in itertools.product(
            maturities,
            moneynesses,
            discount_factors,
        ):
            grid_case_count += 1
            strike = forward / math.exp(x)
            ctx = _context(
                forward=forward,
                discount_factor=discount_factor,
                tau=tau,
            )
            case = _case_identity(
                params=params,
                tau=tau,
                x=x,
                discount_factor=discount_factor,
                forward=forward,
            )
            case["certified_parameter_regime"] = regime_name
            try:
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    robust_call = float(
                        _price(
                            kind=OptionType.CALL,
                            strike=strike,
                            tau=tau,
                            ctx=ctx,
                            params=params,
                            backend="gauss_legendre",
                            quality="robust",
                            recommendation_x=1.3,
                        )
                    )
                    robust_put = float(
                        _price(
                            kind=OptionType.PUT,
                            strike=strike,
                            tau=tau,
                            ctx=ctx,
                            params=params,
                            backend="gauss_legendre",
                            quality="robust",
                            recommendation_x=1.3,
                        )
                    )
                    diagnostics_call = float(
                        _price(
                            kind=OptionType.CALL,
                            strike=strike,
                            tau=tau,
                            ctx=ctx,
                            params=params,
                            backend="gauss_legendre",
                            quality="diagnostics",
                            recommendation_x=1.3,
                        )
                    )
                for item in emitted:
                    warning_counts[
                        f"{item.category.__module__}.{item.category.__name__}"
                    ] += 1
            except Exception as exc:  # pragma: no cover - production evidence path
                fail(
                    "PRICING_EXCEPTION",
                    case,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
                continue

            prices = {
                "robust_call": robust_call,
                "robust_put": robust_put,
                "diagnostics_call": diagnostics_call,
            }
            for name, price in prices.items():
                if not math.isfinite(price):
                    fail("NONFINITE_PRICE", case, price_name=name, value=repr(price))

            discounted_forward = discount_factor * forward
            discounted_strike = discount_factor * strike
            scale = max(discounted_forward, discounted_strike)
            call_lower = max(discounted_forward - discounted_strike, 0.0)
            put_lower = max(discounted_strike - discounted_forward, 0.0)
            bound_tolerance = _bound_tolerance(tolerances, scale=scale)
            bound_deficits = {
                "call_lower": call_lower - robust_call,
                "call_upper": robust_call - discounted_forward,
                "put_lower": put_lower - robust_put,
                "put_upper": robust_put - discounted_strike,
            }
            maximum_bound_deficit = max(0.0, *bound_deficits.values())
            record_maximum("maximum_bound_deficit", maximum_bound_deficit, case)
            if maximum_bound_deficit > bound_tolerance:
                fail(
                    "PRICE_BOUND_FAILURE",
                    case,
                    maximum_deficit=maximum_bound_deficit,
                    tolerance=bound_tolerance,
                    deficits=bound_deficits,
                )

            parity_residual = abs(
                robust_call - robust_put - (discounted_forward - discounted_strike)
            )
            record_maximum("maximum_parity_residual", parity_residual, case)
            parity_tolerance = _parity_tolerance(tolerances, scale=scale)
            if parity_residual > parity_tolerance:
                fail(
                    "PUT_CALL_PARITY_FAILURE",
                    case,
                    residual=parity_residual,
                    tolerance=parity_tolerance,
                )

            diagnostics_difference = abs(diagnostics_call - robust_call)
            record_maximum(
                "maximum_diagnostics_absolute_difference",
                diagnostics_difference,
                case,
            )
            diagnostics_tolerance = _tolerance(
                tolerances,
                prefix="diagnostics",
                scale=discounted_forward,
            )
            if diagnostics_difference > diagnostics_tolerance:
                fail(
                    "DIAGNOSTICS_CONVERGENCE_FAILURE",
                    case,
                    absolute_difference=diagnostics_difference,
                    tolerance=diagnostics_tolerance,
                )

    backend_case_count = 0
    backend_grid = scope["backend_comparison_grid"]
    for parameter_payload in backend_grid["parameter_sets"]:
        params = HestonParams(
            kappa=float(parameter_payload["kappa"]),
            vbar=float(parameter_payload["vbar"]),
            eta=float(parameter_payload["eta"]),
            rho=float(parameter_payload["rho"]),
            v=float(parameter_payload["v0"]),
        )
        for tau, x, discount_factor in itertools.product(
            backend_grid["maturities_years"],
            backend_grid["log_forward_moneyness"],
            backend_grid["discount_factors"],
        ):
            tau = float(tau)
            x = float(x)
            discount_factor = float(discount_factor)
            backend_case_count += 1
            strike = forward / math.exp(x)
            ctx = _context(
                forward=forward,
                discount_factor=discount_factor,
                tau=tau,
            )
            case = _case_identity(
                params=params,
                tau=tau,
                x=x,
                discount_factor=discount_factor,
                forward=forward,
            )
            case["certified_parameter_regime"] = "stable_backend_fixture"
            try:
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    robust_call = float(
                        _price(
                            kind=OptionType.CALL,
                            strike=strike,
                            tau=tau,
                            ctx=ctx,
                            params=params,
                            backend="gauss_legendre",
                            quality="robust",
                            recommendation_x=0.25,
                        )
                    )
                    adaptive_call = float(
                        _price(
                            kind=OptionType.CALL,
                            strike=strike,
                            tau=tau,
                            ctx=ctx,
                            params=params,
                            backend="quad",
                        )
                    )
                for item in emitted:
                    warning_counts[
                        f"{item.category.__module__}.{item.category.__name__}"
                    ] += 1
            except Exception as exc:  # pragma: no cover - production evidence path
                fail(
                    "BACKEND_COMPARISON_EXCEPTION",
                    case,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
                continue
            backend_difference = abs(robust_call - adaptive_call)
            record_maximum(
                "maximum_backend_absolute_difference",
                backend_difference,
                case,
            )
            backend_tolerance = _tolerance(
                tolerances,
                prefix="backend",
                scale=discount_factor * forward,
            )
            if backend_difference > backend_tolerance:
                fail(
                    "BACKEND_AGREEMENT_FAILURE",
                    case,
                    absolute_difference=backend_difference,
                    tolerance=backend_tolerance,
                )

    if warning_counts:
        failures.append(
            {
                "reason_code": "CERTIFIED_GRID_EMITTED_WARNINGS",
                "warning_counts": dict(sorted(warning_counts.items())),
            }
        )

    central_params = HestonParams(kappa=1.5, vbar=0.05, eta=0.5, rho=-0.5, v=0.04)
    scalar_batch_case_count = 0
    scalar_batch_maximum = 0.0
    for kind in (OptionType.CALL, OptionType.PUT):
        for tau in maturities:
            ctx = _context(forward=forward, discount_factor=0.9, tau=tau)
            strikes = np.asarray(
                [forward / math.exp(x) for x in moneynesses],
                dtype=np.float64,
            )
            batch = np.asarray(
                _price(
                    kind=kind,
                    strike=strikes,
                    tau=tau,
                    ctx=ctx,
                    params=central_params,
                    backend="gauss_legendre",
                    quality="robust",
                    recommendation_x=1.3,
                ),
                dtype=np.float64,
            )
            scalar = np.asarray(
                [
                    _price(
                        kind=kind,
                        strike=float(strike),
                        tau=tau,
                        ctx=ctx,
                        params=central_params,
                        backend="gauss_legendre",
                        quality="robust",
                        recommendation_x=1.3,
                    )
                    for strike in strikes
                ],
                dtype=np.float64,
            )
            difference = float(np.max(np.abs(batch - scalar)))
            scalar_batch_maximum = max(scalar_batch_maximum, difference)
            scalar_batch_case_count += len(strikes)
            tolerance = float(tolerances["scalar_batch_absolute_price"]) + float(
                tolerances["scalar_batch_relative_price"]
            ) * float(np.max(np.abs(scalar)))
            if difference > tolerance:
                failures.append(
                    {
                        "reason_code": "SCALAR_BATCH_CONSISTENCY_FAILURE",
                        "option_right": kind.value,
                        "maturity_years": tau,
                        "maximum_absolute_difference": difference,
                        "tolerance": tolerance,
                    }
                )

    scale_case_count = 0
    scale_maximum = 0.0
    for kind in (OptionType.CALL, OptionType.PUT):
        for x in moneynesses:
            normalized_prices: list[float] = []
            for scale_forward in market["forward_scale_checks"]:
                scale_forward = float(scale_forward)
                tau = 1.0
                discount_factor = 0.9
                ctx = _context(
                    forward=scale_forward,
                    discount_factor=discount_factor,
                    tau=tau,
                )
                strike = scale_forward / math.exp(x)
                price = float(
                    _price(
                        kind=kind,
                        strike=strike,
                        tau=tau,
                        ctx=ctx,
                        params=central_params,
                        backend="gauss_legendre",
                        quality="robust",
                    )
                )
                normalized_prices.append(price / (discount_factor * scale_forward))
                scale_case_count += 1
            difference = max(normalized_prices) - min(normalized_prices)
            scale_maximum = max(scale_maximum, difference)
            if difference > float(
                tolerances["scale_homogeneity_absolute_normalized_price"]
            ):
                failures.append(
                    {
                        "reason_code": "FORWARD_SCALE_HOMOGENEITY_FAILURE",
                        "option_right": kind.value,
                        "log_forward_moneyness": x,
                        "normalized_price_range": difference,
                    }
                )

    return {
        "schema_version": "opl_heston_numerical_grid_evidence.v2",
        "status": "PASS" if not failures else "FAIL",
        "certified_grid": {
            "parameter_regime_count": len(scope["certified_parameter_regimes"]),
            "parameter_corner_count": len(parameter_cases),
            "market_boundary_count_per_parameter_corner": (
                len(maturities) * len(moneynesses) * len(discount_factors)
            ),
            "case_count": grid_case_count,
            "pricing_operation_count": (grid_case_count * 3 + backend_case_count * 2),
        },
        "summaries": {
            "finite_outputs": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"] == "NONFINITE_PRICE" for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": grid_case_count * 3,
            },
            "bounds": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"] == "PRICE_BOUND_FAILURE"
                        for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": grid_case_count * 2,
                **maximums["maximum_bound_deficit"],
            },
            "parity": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"] == "PUT_CALL_PARITY_FAILURE"
                        for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": grid_case_count,
                **maximums["maximum_parity_residual"],
            },
            "backend_comparison": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"]
                        in {
                            "BACKEND_AGREEMENT_FAILURE",
                            "BACKEND_COMPARISON_EXCEPTION",
                        }
                        for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": backend_case_count,
                **maximums["maximum_backend_absolute_difference"],
            },
            "diagnostics_convergence": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"] == "DIAGNOSTICS_CONVERGENCE_FAILURE"
                        for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": grid_case_count,
                **maximums["maximum_diagnostics_absolute_difference"],
            },
            "scalar_batch_consistency": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"] == "SCALAR_BATCH_CONSISTENCY_FAILURE"
                        for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": scalar_batch_case_count,
                "maximum_absolute_difference": scalar_batch_maximum,
            },
            "forward_scale_homogeneity": {
                "status": (
                    "PASS"
                    if not any(
                        item["reason_code"] == "FORWARD_SCALE_HOMOGENEITY_FAILURE"
                        for item in failures
                    )
                    else "FAIL"
                ),
                "case_count": scale_case_count,
                "maximum_normalized_price_range": scale_maximum,
            },
            "certified_grid_warnings": {
                "status": "PASS" if not warning_counts else "FAIL",
                "warning_counts": dict(sorted(warning_counts.items())),
            },
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text(encoding="utf-8"))
    result = run_probe(scope)
    _write_json(args.output, result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
