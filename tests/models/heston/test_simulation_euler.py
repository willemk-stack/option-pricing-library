import math
from types import SimpleNamespace

import numpy as np
import pytest

from option_pricing.models.heston.params import HestonParams
from option_pricing.models.heston.simulation import (
    HestonPathSimulator,
    simulate_heston_paths,
    simulate_heston_terminal,
)
from option_pricing.models.heston.simulation.euler import simulate_heston_euler_paths
from option_pricing.monte_carlo import MCConfig, RandomConfig
from option_pricing.types import MarketData


@pytest.fixture
def heston_params() -> HestonParams:
    return HestonParams(
        kappa=2.0,
        vbar=0.04,
        eta=0.5,
        rho=-0.7,
        v=0.04,
    )


def _ctx():
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context()


def test_simulate_heston_euler_paths_returns_expected_shapes(
    heston_params: HestonParams,
) -> None:
    n_paths = 3
    n_steps = 4
    tau = 1.0
    x0 = 100.0

    shocks = np.zeros((n_paths, n_steps, 2), dtype=np.float64)
    drift = np.zeros(n_steps, dtype=np.float64)

    result = simulate_heston_euler_paths(
        params=heston_params,
        x0=x0,
        tau=tau,
        n_steps=n_steps,
        shocks=shocks,
        log_drift_increments=drift,
    )

    assert result.spot_paths.shape == (n_paths, n_steps + 1)
    assert result.var_paths.shape == (n_paths, n_steps + 1)
    assert result.dt == pytest.approx(tau / n_steps)

    np.testing.assert_allclose(result.spot_paths[:, 0], x0)
    np.testing.assert_allclose(result.var_paths[:, 0], heston_params.v)


def test_zero_variance_zero_drift_produces_flat_spot_paths() -> None:
    params = HestonParams(
        kappa=1.0,
        vbar=0.0,
        eta=0.0,
        rho=0.0,
        v=0.0,
    )

    n_paths = 5
    n_steps = 6
    x0 = 123.45

    shocks = np.random.default_rng(123).normal(size=(n_paths, n_steps, 2))
    drift = np.zeros(n_steps, dtype=np.float64)

    result = simulate_heston_euler_paths(
        params=params,
        x0=x0,
        tau=1.0,
        n_steps=n_steps,
        shocks=shocks,
        log_drift_increments=drift,
    )

    np.testing.assert_allclose(result.spot_paths, x0)
    np.testing.assert_allclose(result.var_paths, 0.0)


def test_one_step_matches_manual_full_truncation_update(
    heston_params: HestonParams,
) -> None:
    x0 = 100.0
    tau = 0.25
    n_steps = 1
    dt = tau / n_steps

    z_x = 0.3
    z_v = -0.2
    log_drift = 0.01

    shocks = np.array([[[z_x, z_v]]], dtype=np.float64)
    drift = np.array([log_drift], dtype=np.float64)

    result = simulate_heston_euler_paths(
        params=heston_params,
        x0=x0,
        tau=tau,
        n_steps=n_steps,
        shocks=shocks,
        log_drift_increments=drift,
    )

    v0 = heston_params.v
    v_pos = max(v0, 0.0)

    expected_v1 = (
        v0
        + heston_params.kappa * (heston_params.vbar - v_pos) * dt
        + heston_params.eta * math.sqrt(v_pos * dt) * z_v
    )

    expected_log_x1 = (
        math.log(x0) + log_drift - 0.5 * v_pos * dt + math.sqrt(v_pos * dt) * z_x
    )

    assert result.var_paths[0, 1] == pytest.approx(expected_v1)
    assert result.spot_paths[0, 1] == pytest.approx(math.exp(expected_log_x1))


def test_default_log_drift_increments_means_zero_drift(
    heston_params: HestonParams,
) -> None:
    n_paths = 2
    n_steps = 3
    x0 = 100.0

    shocks = np.zeros((n_paths, n_steps, 2), dtype=np.float64)

    result_default = simulate_heston_euler_paths(
        params=heston_params,
        x0=x0,
        tau=1.0,
        n_steps=n_steps,
        shocks=shocks,
        log_drift_increments=None,
    )

    result_explicit_zero = simulate_heston_euler_paths(
        params=heston_params,
        x0=x0,
        tau=1.0,
        n_steps=n_steps,
        shocks=shocks,
        log_drift_increments=np.zeros(n_steps, dtype=np.float64),
    )

    np.testing.assert_allclose(
        result_default.spot_paths, result_explicit_zero.spot_paths
    )
    np.testing.assert_allclose(result_default.var_paths, result_explicit_zero.var_paths)


def test_simulate_heston_paths_returns_expected_path_shape(
    heston_params: HestonParams,
) -> None:
    n_paths = 6
    n_steps = 8

    paths = simulate_heston_paths(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=n_steps,
        cfg=MCConfig(n_paths=n_paths, random=RandomConfig(seed=17)),
        scheme="euler_full_truncation",
    )

    assert paths.shape == (n_paths, n_steps + 1)
    np.testing.assert_allclose(paths[:, 0], 100.0)
    assert np.all(np.isfinite(paths))


def test_simulate_heston_terminal_returns_expected_terminal_shape(
    heston_params: HestonParams,
) -> None:
    terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=23)),
        scheme="euler_full_truncation",
    )

    assert terminal.shape == (6,)
    assert np.all(np.isfinite(terminal))


def test_heston_path_simulator_is_deterministic_with_same_seed(
    heston_params: HestonParams,
) -> None:
    simulator = HestonPathSimulator(
        params=heston_params,
        n_steps=10,
        scheme="euler_full_truncation",
    )
    cfg = MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=31))

    paths_a = simulator.simulate_paths(ctx=_ctx(), tau=0.75, cfg=cfg)
    paths_b = simulator.simulate_paths(ctx=_ctx(), tau=0.75, cfg=cfg)

    np.testing.assert_allclose(paths_a, paths_b)


@pytest.mark.parametrize(
    ("tau", "n_steps", "x0", "error_message"),
    [
        (-1.0, 10, 100.0, "tau must be finite and non-negative"),
        (np.inf, 10, 100.0, "tau must be finite and non-negative"),
        (1.0, 0, 100.0, "n_steps must be positive"),
        (1.0, 10, 0.0, "x0 must be finite and positive"),
        (1.0, 10, -100.0, "x0 must be finite and positive"),
        (1.0, 10, np.inf, "x0 must be finite and positive"),
    ],
)
def test_invalid_scalar_inputs_raise(
    heston_params: HestonParams,
    tau: float,
    n_steps: int,
    x0: float,
    error_message: str,
) -> None:
    shocks = np.zeros((2, max(n_steps, 1), 2), dtype=np.float64)

    with pytest.raises(ValueError, match=error_message):
        simulate_heston_euler_paths(
            params=heston_params,
            x0=x0,
            tau=tau,
            n_steps=n_steps,
            shocks=shocks,
            log_drift_increments=np.zeros(max(n_steps, 1), dtype=np.float64),
        )


@pytest.mark.parametrize(
    "shocks",
    [
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 3, 1), dtype=np.float64),
        np.zeros((2, 3, 3), dtype=np.float64),
    ],
)
def test_invalid_shock_shape_raises(
    heston_params: HestonParams, shocks: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shocks must"):
        simulate_heston_euler_paths(
            params=heston_params,
            x0=100.0,
            tau=1.0,
            n_steps=3,
            shocks=shocks,
            log_drift_increments=np.zeros(3, dtype=np.float64),
        )


def test_shock_step_count_must_match_n_steps(heston_params: HestonParams) -> None:
    shocks = np.zeros((2, 4, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="shocks has 4 steps, expected 3"):
        simulate_heston_euler_paths(
            params=heston_params,
            x0=100.0,
            tau=1.0,
            n_steps=3,
            shocks=shocks,
            log_drift_increments=np.zeros(3, dtype=np.float64),
        )


def test_log_drift_increments_must_have_one_value_per_step(
    heston_params: HestonParams,
) -> None:
    shocks = np.zeros((2, 3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="log_drift_increments must have shape"):
        simulate_heston_euler_paths(
            params=heston_params,
            x0=100.0,
            tau=1.0,
            n_steps=3,
            shocks=shocks,
            log_drift_increments=np.zeros(2, dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("tau", "n_steps", "error_message"),
    [
        (0.0, 4, "tau must be positive"),
        (-0.5, 4, "tau must be positive"),
        (1.0, 0, "n_steps must be positive"),
    ],
)
def test_simulators_validate_bad_tau_and_n_steps(
    heston_params: HestonParams,
    tau: float,
    n_steps: int,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        simulate_heston_paths(
            ctx=_ctx(),
            tau=tau,
            params=heston_params,
            n_steps=n_steps,
            cfg=MCConfig(n_paths=2, random=RandomConfig(seed=1)),
        )


def test_simulate_heston_euler_paths_validates_bad_initial_variance_param() -> None:
    with pytest.raises(ValueError, match="v0 must be finite and non-negative"):
        simulate_heston_euler_paths(
            params=SimpleNamespace(
                kappa=1.0,
                vbar=0.04,
                eta=0.5,
                rho=0.0,
                v=-0.01,
            ),
            x0=100.0,
            tau=1.0,
            n_steps=2,
            shocks=np.zeros((1, 2, 2), dtype=np.float64),
            log_drift_increments=np.zeros(2, dtype=np.float64),
        )


def test_full_truncation_keeps_variance_non_negative() -> None:
    params = HestonParams(
        kappa=1.0,
        vbar=0.0,
        eta=10.0,
        rho=0.0,
        v=0.01,
    )

    shocks = np.array(
        [
            [
                [0.0, -10.0],
                [0.0, 10.0],
            ]
        ],
        dtype=np.float64,
    )

    result = simulate_heston_euler_paths(
        params=params,
        x0=100.0,
        tau=1.0,
        n_steps=2,
        shocks=shocks,
        log_drift_increments=np.zeros(2, dtype=np.float64),
    )

    assert np.all(np.isfinite(result.spot_paths))
    assert np.all(np.isfinite(result.var_paths))
    assert np.all(result.var_paths >= 0.0)


def test_euler_metadata_tracks_negative_variance_proposal_rate() -> None:
    params = HestonParams(
        kappa=1.0,
        vbar=0.0,
        eta=10.0,
        rho=0.0,
        v=0.01,
    )

    shocks = np.array(
        [
            [[0.0, -1.0]],
            [[0.0, 0.2]],
        ],
        dtype=np.float64,
    )

    result = simulate_heston_euler_paths(
        params=params,
        x0=100.0,
        tau=1.0,
        n_steps=1,
        shocks=shocks,
        log_drift_increments=np.zeros(1, dtype=np.float64),
    )

    assert np.all(result.var_paths >= 0.0)
    assert result.metadata["negative_variance_proposal_rate"] == pytest.approx(0.5)
