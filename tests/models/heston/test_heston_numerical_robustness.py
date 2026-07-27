from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import option_pricing.models.heston.calibration.calibrate as calibrate_module
import option_pricing.models.heston.calibration.objective as objective_module
import option_pricing.models.heston.calibration.validation as validation_module
from option_pricing.exceptions import NoConvergenceError
from option_pricing.models.heston.calibration import (
    HestonCalibrationBounds,
    HestonNumericalEvaluationError,
    validate_heston_calibration_solution,
)
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.charfunc import _cui_char_fn_and_param_grad
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.types import MarketData


def _params(*, kappa: float = 1.5) -> HestonParams:
    return HestonParams(kappa=kappa, vbar=0.04, eta=0.45, rho=-0.6, v=0.05)


def _quotes() -> HestonQuoteSet:
    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.01, dividend_yield=0.0),
        strike=np.array([90.0, 100.0, 110.0], dtype=np.float64),
        expiry=np.array([0.5, 1.0, 1.0], dtype=np.float64),
        is_call=np.array([True, True, False], dtype=np.bool_),
        mid=np.array([12.0, 8.0, 11.0], dtype=np.float64),
    )


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=30.0, n_panels=3, nodes_per_panel=6)


def test_overflowing_trial_is_structured_without_runtime_warning() -> None:
    objective = HestonObjective(
        quotes=_quotes(),
        objective_type="price_rmse",
        parameter_transform="unconstrained",
        quad_cfg=_quad_cfg(),
    )
    raw = _params().transform_to_unconstrained()
    raw[0] = 1.0e308
    previous_errstate = np.geterr()

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(HestonNumericalEvaluationError) as excinfo:
            objective.residual(raw)

    assert excinfo.value.evaluation_stage == "objective_residual"
    assert excinfo.value.category in {"overflow", "nonfinite_result"}
    assert excinfo.value.parameter_vector is not None
    assert not [item for item in emitted if issubclass(item.category, RuntimeWarning)]
    assert np.geterr() == previous_errstate


def test_analytic_jacobian_divide_by_zero_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = HestonObjective(
        quotes=_quotes(),
        objective_type="price_rmse",
        quad_cfg=_quad_cfg(),
    )

    def singular_price_jacobian(*_args: object, **_kwargs: object) -> None:
        np.divide(
            np.ones(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )

    monkeypatch.setattr(
        objective_module,
        "_price_and_jac_heston_quotes",
        singular_price_jacobian,
    )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(HestonNumericalEvaluationError) as excinfo:
            objective.jac(_params().transform_to_unconstrained())

    assert excinfo.value.evaluation_stage == "objective_jacobian"
    assert excinfo.value.category == "divide_by_zero"
    assert not [item for item in emitted if issubclass(item.category, RuntimeWarning)]


def test_optimizer_callbacks_never_return_nonfinite_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = HestonObjective(
        quotes=_quotes(),
        objective_type="price_rmse",
        quad_cfg=_quad_cfg(),
    )
    raw = _params().transform_to_unconstrained()

    monkeypatch.setattr(
        objective_module,
        "_price_heston_quotes",
        lambda *_args, **_kwargs: np.full(3, np.nan, dtype=np.float64),
    )
    with pytest.raises(HestonNumericalEvaluationError, match="model_prices"):
        objective.residual(raw)

    monkeypatch.setattr(
        objective_module,
        "_price_and_jac_heston_quotes",
        lambda *_args, **_kwargs: (
            np.ones(3, dtype=np.float64),
            np.full((3, 5), np.inf, dtype=np.float64),
        ),
    )
    with pytest.raises(HestonNumericalEvaluationError, match="price_jacobian"):
        objective.jac(raw)


def test_failed_numerical_seed_retains_context_and_other_seed_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = [_params(kappa=1.0), _params(kappa=2.0)]
    calls = {"count": 0}

    def fake_calibrate_heston(
        **kwargs: object,
    ) -> tuple[HestonParams, OptimizeResult]:
        calls["count"] += 1
        seed = kwargs["x0_params"]
        assert isinstance(seed, HestonParams)
        if calls["count"] == 1:
            raise HestonNumericalEvaluationError(
                evaluation_stage="objective_jacobian",
                category="singular_denominator",
                message="synthetic A2 singularity",
                maturity=1.0,
                probability_index=1,
                parameter_vector=seed.as_array(),
                failing_expression="A2_scaled",
            )
        return seed, OptimizeResult(
            x=seed.transform_to_unconstrained(),
            success=True,
            cost=0.25,
            message="ok",
        )

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    result = calibrate_module.calibrate_heston_multistart(
        quotes=_quotes(),
        objective_type="price_rmse",
        seeds=seeds,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    failure = result.failed_runs[0].numerical_failure
    assert failure is not None
    assert failure.seed_index == 0
    assert failure.maturity == 1.0
    assert failure.probability_index == 1
    assert failure.failing_expression == "A2_scaled"


def test_all_numerically_failed_seeds_preserve_all_failed_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_seed(**kwargs: object) -> None:
        seed = kwargs["x0_params"]
        assert isinstance(seed, HestonParams)
        raise HestonNumericalEvaluationError(
            evaluation_stage="objective_residual",
            category="overflow",
            message="synthetic overflow",
            parameter_vector=seed.as_array(),
            failing_expression="characteristic_exponent",
        )

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fail_seed)

    with pytest.raises(NoConvergenceError, match=r"all 2 seed\(s\) failed"):
        calibrate_module.calibrate_heston_multistart(
            quotes=_quotes(),
            objective_type="price_rmse",
            seeds=[_params(kappa=1.0), _params(kappa=2.0)],
        )


def test_scaled_cui_expressions_avoid_balanced_exponential_overflow() -> None:
    params = HestonParams(
        kappa=20.0,
        vbar=0.04,
        eta=0.5,
        rho=-0.5,
        v=0.04,
    )
    frequencies = np.array([0.2, 1.0, 5.0], dtype=np.float64)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        phi, jac = _cui_char_fn_and_param_grad(
            frequencies,
            100.0,
            params,
        )

    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(jac))
    assert not [item for item in emitted if issubclass(item.category, RuntimeWarning)]


def test_authoritative_selected_solution_validation_is_clean() -> None:
    validation = validate_heston_calibration_solution(
        _quotes(),
        _params(),
        bounds=HestonCalibrationBounds(),
        quad_cfg=_quad_cfg(),
    )

    assert validation.analytic_jacobian_validated
    assert np.all(np.isfinite(validation.model_prices))
    assert validation.price_jacobian is not None
    assert np.all(np.isfinite(validation.price_jacobian))
    assert np.isfinite(validation.price_rmse)
    assert np.isfinite(validation.max_abs_price_error)


def test_authoritative_validation_promotes_runtime_warning_to_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning_price_jacobian(*_args: object, **_kwargs: object) -> None:
        warnings.warn("overflow encountered in exp", RuntimeWarning, stacklevel=1)

    monkeypatch.setattr(
        validation_module,
        "_price_and_jac_heston_quotes",
        warning_price_jacobian,
    )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(HestonNumericalEvaluationError) as excinfo:
            validate_heston_calibration_solution(
                _quotes(),
                _params(),
                bounds=HestonCalibrationBounds(),
                quad_cfg=_quad_cfg(),
            )

    assert excinfo.value.evaluation_stage == "selected_solution_validation"
    assert excinfo.value.category == "overflow"
    assert not [item for item in emitted if issubclass(item.category, RuntimeWarning)]


def test_optimizer_success_is_rejected_when_selected_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_least_squares(**kwargs: object) -> OptimizeResult:
        return OptimizeResult(
            x=np.asarray(kwargs["x0"], dtype=np.float64),
            success=True,
            cost=0.25,
            optimality=1.0e-8,
            message="optimizer converged",
        )

    def fail_selected_validation(*_args: object, **_kwargs: object) -> None:
        raise HestonNumericalEvaluationError(
            evaluation_stage="selected_solution_validation",
            category="nonfinite_result",
            message="synthetic nonfinite selected price",
            failing_expression="selected_model_prices",
        )

    monkeypatch.setattr(calibrate_module, "least_squares", fake_least_squares)
    monkeypatch.setattr(
        calibrate_module,
        "validate_heston_calibration_solution",
        fail_selected_validation,
    )

    with pytest.raises(
        HestonNumericalEvaluationError,
        match="selected_model_prices",
    ):
        calibrate_module.calibrate_heston(
            _quotes(),
            objective_type="price_rmse",
            x0_params=_params(),
            quad_cfg=_quad_cfg(),
            use_analytic_jac=False,
            max_nfev=1,
        )


def test_authoritative_validation_rejects_selected_params_outside_bounds() -> None:
    with pytest.raises(HestonNumericalEvaluationError) as excinfo:
        validate_heston_calibration_solution(
            _quotes(),
            _params(kappa=21.0),
            bounds=HestonCalibrationBounds(),
            quad_cfg=_quad_cfg(),
            require_analytic_jacobian=False,
        )

    assert excinfo.value.category == "outside_bounds"
    assert excinfo.value.failing_expression == "bounds[kappa]"
