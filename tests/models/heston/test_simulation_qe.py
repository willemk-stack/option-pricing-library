import numpy as np
import pytest

import option_pricing.models.heston.simulation.engine as engine_module
import option_pricing.models.heston.simulation.qe as qe_module
from option_pricing.models.heston.params import HestonParams
from option_pricing.models.heston.simulation import (
    HestonPathSimulator,
    HestonTerminalSimulator,
    simulate_heston_paths,
    simulate_heston_terminal,
)
from option_pricing.models.heston.simulation.qe import (
    _psi,
    _v_timestep_exponential,
    _v_timestep_qe,
    _v_timestep_quadratic,
    _x_timestep_qe,
    simulate_heston_qe_paths,
    simulate_heston_qe_terminal,
)
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


def _qe_shocks(n_paths: int, n_steps: int) -> np.ndarray:
    shocks = np.zeros((n_paths, n_steps, 3), dtype=np.float64)
    shocks[:, :, 0] = 0.1
    shocks[:, :, 1] = 0.5
    shocks[:, :, 2] = -0.2
    return shocks


def test_simulate_heston_qe_paths_returns_standard_result_shape(
    heston_params: HestonParams,
) -> None:
    n_paths = 3
    n_steps = 4
    tau = 1.0
    x0 = 100.0

    result = simulate_heston_qe_paths(
        params=heston_params,
        x0=x0,
        tau=tau,
        n_steps=n_steps,
        shocks=_qe_shocks(n_paths=n_paths, n_steps=n_steps),
        log_drift_increments=np.zeros(n_steps, dtype=np.float64),
    )

    assert result.spot_paths.shape == (n_paths, n_steps + 1)
    assert result.var_paths.shape == (n_paths, n_steps + 1)
    assert result.dt == pytest.approx(tau / n_steps)

    np.testing.assert_allclose(result.spot_paths[:, 0], x0)
    np.testing.assert_allclose(result.var_paths[:, 0], heston_params.v)
    assert np.all(np.isfinite(result.spot_paths))
    assert np.all(np.isfinite(result.var_paths))


def test_simulate_heston_qe_paths_accepts_pathwise_log_drift(
    heston_params: HestonParams,
) -> None:
    n_paths = 2
    n_steps = 3
    shocks = _qe_shocks(n_paths=n_paths, n_steps=n_steps)
    drift = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.02, 0.02],
        ],
        dtype=np.float64,
    )

    result = simulate_heston_qe_paths(
        params=heston_params,
        x0=100.0,
        tau=1.0,
        n_steps=n_steps,
        shocks=shocks,
        log_drift_increments=drift,
    )

    np.testing.assert_allclose(result.var_paths[0], result.var_paths[1])
    assert result.spot_paths[1, -1] > result.spot_paths[0, -1]


@pytest.mark.parametrize(
    "shocks",
    [
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 3, 2), dtype=np.float64),
        np.zeros((2, 3, 4), dtype=np.float64),
    ],
)
def test_simulate_heston_qe_paths_validates_shock_shape(
    heston_params: HestonParams,
    shocks: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="shocks must"):
        simulate_heston_qe_paths(
            params=heston_params,
            x0=100.0,
            tau=1.0,
            n_steps=3,
            shocks=shocks,
            log_drift_increments=np.zeros(3, dtype=np.float64),
        )


def test_simulate_heston_qe_paths_validates_shock_step_count(
    heston_params: HestonParams,
) -> None:
    shocks = np.zeros((2, 4, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="shocks has 4 steps, expected 3"):
        simulate_heston_qe_paths(
            params=heston_params,
            x0=100.0,
            tau=1.0,
            n_steps=3,
            shocks=shocks,
            log_drift_increments=np.zeros(3, dtype=np.float64),
        )


def test_simulate_heston_paths_supports_quadratic_exponential_scheme(
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
        scheme="quadratic_exponential",
    )

    assert paths.shape == (n_paths, n_steps + 1)
    np.testing.assert_allclose(paths[:, 0], 100.0)
    assert np.all(np.isfinite(paths))


def test_heston_path_simulator_defaults_to_quadratic_exponential(
    heston_params: HestonParams,
) -> None:
    cfg = MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=31))

    default_paths = HestonPathSimulator(
        params=heston_params,
        n_steps=10,
    ).simulate_paths(ctx=_ctx(), tau=0.75, cfg=cfg)
    explicit_paths = HestonPathSimulator(
        params=heston_params,
        n_steps=10,
        scheme="quadratic_exponential",
    ).simulate_paths(ctx=_ctx(), tau=0.75, cfg=cfg)

    np.testing.assert_allclose(default_paths, explicit_paths)


def test_heston_terminal_simulator_defaults_to_quadratic_exponential(
    heston_params: HestonParams,
) -> None:
    cfg = MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=19))

    default_terminal = HestonTerminalSimulator(
        params=heston_params,
        n_steps=10,
    ).simulate_terminal(ctx=_ctx(), tau=0.75, cfg=cfg)
    explicit_terminal = HestonTerminalSimulator(
        params=heston_params,
        n_steps=10,
        scheme="quadratic_exponential",
    ).simulate_terminal(ctx=_ctx(), tau=0.75, cfg=cfg)

    np.testing.assert_allclose(default_terminal, explicit_terminal)


def test_simulation_helpers_default_to_quadratic_exponential(
    heston_params: HestonParams,
) -> None:
    default_paths = simulate_heston_paths(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=MCConfig(n_paths=6, random=RandomConfig(seed=17)),
    )
    explicit_paths = simulate_heston_paths(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=MCConfig(n_paths=6, random=RandomConfig(seed=17)),
        scheme="quadratic_exponential",
    )

    np.testing.assert_allclose(default_paths, explicit_paths)

    default_terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=23)),
    )
    explicit_terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=23)),
        scheme="quadratic_exponential",
    )

    np.testing.assert_allclose(default_terminal, explicit_terminal)


def test_heston_path_simulator_qe_is_deterministic_with_same_seed(
    heston_params: HestonParams,
) -> None:
    simulator = HestonPathSimulator(
        params=heston_params,
        n_steps=10,
        scheme="quadratic_exponential",
    )
    cfg = MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=31))

    paths_a = simulator.simulate_paths(ctx=_ctx(), tau=0.75, cfg=cfg)
    paths_b = simulator.simulate_paths(ctx=_ctx(), tau=0.75, cfg=cfg)

    np.testing.assert_allclose(paths_a, paths_b)


def test_simulate_heston_terminal_supports_quadratic_exponential_scheme(
    heston_params: HestonParams,
) -> None:
    terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=23)),
        scheme="quadratic_exponential",
    )

    assert terminal.shape == (6,)
    assert np.all(np.isfinite(terminal))


def test_simulate_heston_qe_supports_zero_vol_of_vol() -> None:
    terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=HestonParams(kappa=1.5, vbar=0.04, eta=0.0, rho=0.0, v=0.04),
        n_steps=8,
        cfg=MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=23)),
        scheme="quadratic_exponential",
    )

    assert terminal.shape == (6,)
    assert np.all(np.isfinite(terminal))


def test_v_timestep_qe_uses_quadratic_branch_at_cutoff(
    heston_params: HestonParams,
) -> None:
    v_t = np.array([heston_params.v], dtype=np.float64)
    z_v = np.array([0.25], dtype=np.float64)
    u_v = np.array([0.9], dtype=np.float64)
    dt = 0.5
    psi_c = float(_psi(v_t=v_t, params=heston_params, dt=dt)[0])

    v_next = _v_timestep_qe(
        v_t=v_t,
        params=heston_params,
        z_v_j=z_v,
        u_v_j=u_v,
        dt=dt,
        psi_c=psi_c,
    )
    expected = _v_timestep_quadratic(
        v_t=v_t,
        params=heston_params,
        z_v_j=z_v,
        dt=dt,
    )

    np.testing.assert_allclose(v_next, expected)


def test_v_timestep_qe_switches_to_exponential_just_above_cutoff(
    heston_params: HestonParams,
) -> None:
    v_t = np.array([heston_params.v], dtype=np.float64)
    z_v = np.array([0.25], dtype=np.float64)
    u_v = np.array([0.9], dtype=np.float64)
    dt = 0.5
    psi = float(_psi(v_t=v_t, params=heston_params, dt=dt)[0])

    v_next = _v_timestep_qe(
        v_t=v_t,
        params=heston_params,
        z_v_j=z_v,
        u_v_j=u_v,
        dt=dt,
        psi_c=psi - 1e-6,
    )
    expected = _v_timestep_exponential(
        v_t=v_t,
        params=heston_params,
        u_v_j=u_v,
        dt=dt,
    )

    np.testing.assert_allclose(v_next, expected)


def test_x_timestep_qe_martingale_correction_moves_mean_toward_unity() -> None:
    params = HestonParams(
        kappa=2.0,
        vbar=0.04,
        eta=0.5,
        rho=0.9,
        v=0.04,
    )

    rng = np.random.default_rng(123)
    n_paths = 150_000
    dt = 0.75

    v_t = np.full(n_paths, params.v, dtype=np.float64)
    z_v = rng.standard_normal(n_paths)
    u_v = rng.random(n_paths)
    z_x = rng.standard_normal(n_paths)

    v_next = _v_timestep_qe(
        v_t=v_t,
        params=params,
        z_v_j=z_v,
        u_v_j=u_v,
        dt=dt,
    )

    log_x_corrected = _x_timestep_qe(
        log_x_t=np.zeros(n_paths, dtype=np.float64),
        v_t=v_t,
        v_next=v_next,
        params=params,
        z_x_j=z_x,
        dt=dt,
        martingale_correction=True,
    )
    log_x_uncorrected = _x_timestep_qe(
        log_x_t=np.zeros(n_paths, dtype=np.float64),
        v_t=v_t,
        v_next=v_next,
        params=params,
        z_x_j=z_x,
        dt=dt,
        martingale_correction=False,
    )

    corrected_mean = float(np.exp(log_x_corrected).mean())
    uncorrected_mean = float(np.exp(log_x_uncorrected).mean())

    assert abs(corrected_mean - 1.0) < abs(uncorrected_mean - 1.0)
    assert abs(corrected_mean - 1.0) < 5e-4


def test_heston_qe_terminal_forward_matches_theoretical_forward_within_ci() -> None:
    params = HestonParams(
        kappa=1.5,
        vbar=0.04,
        eta=0.8,
        rho=-0.7,
        v=0.05,
    )
    tau = 1.5
    n_paths = 12_000
    ci_multiplier = 4.0
    small_abs_buffer = 0.05

    terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=tau,
        params=params,
        n_steps=32,
        cfg=MCConfig(
            n_paths=n_paths,
            antithetic=True,
            random=RandomConfig(seed=1234),
        ),
        scheme="quadratic_exponential",
    )

    expected_forward = _ctx().fwd(tau)
    sample_mean = float(np.mean(terminal))
    sample_std = float(np.std(terminal, ddof=1))
    standard_error = sample_std / np.sqrt(n_paths)
    error = abs(sample_mean - expected_forward)

    # NOTE: This is a martingale/forward-consistency regression guard, not a
    # statistical power test, so the CI is intentionally generous for CI
    # stability while still catching broken drift or correction wiring.
    assert error <= ci_multiplier * standard_error + small_abs_buffer


def test_simulate_heston_qe_terminal_matches_full_path_terminal(
    heston_params: HestonParams,
) -> None:
    rng = np.random.default_rng(123)
    shocks = np.empty((5, 7, 3), dtype=np.float64)
    shocks[:, :, 0] = rng.standard_normal((5, 7))
    shocks[:, :, 1] = rng.random((5, 7))
    shocks[:, :, 2] = rng.standard_normal((5, 7))
    drift = rng.normal(scale=0.01, size=(5, 7))

    path_result = simulate_heston_qe_paths(
        params=heston_params,
        x0=100.0,
        tau=1.25,
        n_steps=7,
        shocks=shocks,
        log_drift_increments=drift,
    )
    terminal = simulate_heston_qe_terminal(
        params=heston_params,
        x0=100.0,
        tau=1.25,
        n_steps=7,
        shocks=shocks,
        log_drift_increments=drift,
    )

    np.testing.assert_allclose(terminal, path_result.spot_paths[:, -1])


def test_simulate_heston_qe_terminal_matches_full_path_terminal_when_eta_zero() -> None:
    params = HestonParams(kappa=1.5, vbar=0.04, eta=0.0, rho=0.0, v=0.08)
    rng = np.random.default_rng(321)
    shocks = np.empty((4, 6, 3), dtype=np.float64)
    shocks[:, :, 0] = rng.standard_normal((4, 6))
    shocks[:, :, 1] = rng.random((4, 6))
    shocks[:, :, 2] = rng.standard_normal((4, 6))
    drift = rng.normal(scale=0.01, size=6)

    path_result = simulate_heston_qe_paths(
        params=params,
        x0=100.0,
        tau=0.75,
        n_steps=6,
        shocks=shocks,
        log_drift_increments=drift,
    )
    terminal = simulate_heston_qe_terminal(
        params=params,
        x0=100.0,
        tau=0.75,
        n_steps=6,
        shocks=shocks,
        log_drift_increments=drift,
    )

    np.testing.assert_allclose(terminal, path_result.spot_paths[:, -1])


def test_simulate_heston_qe_terminal_avoids_full_path_initializer(
    heston_params: HestonParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_initialize_paths(**_kwargs):
        raise AssertionError("full-path allocation should not be used")

    monkeypatch.setattr(qe_module, "_initialize_paths", _fail_initialize_paths)

    terminal = simulate_heston_qe_terminal(
        params=heston_params,
        x0=100.0,
        tau=1.0,
        n_steps=4,
        shocks=_qe_shocks(n_paths=2, n_steps=4),
        log_drift_increments=np.zeros(4, dtype=np.float64),
    )

    assert terminal.shape == (2,)
    assert np.all(np.isfinite(terminal))


@pytest.mark.parametrize("scheme", ["euler_full_truncation", "quadratic_exponential"])
def test_simulate_heston_terminal_does_not_delegate_to_path_simulator(
    heston_params: HestonParams,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    def _fail_simulate_paths(self, *, ctx, tau, cfg):
        raise AssertionError(
            "terminal simulation should not delegate to path simulation"
        )

    monkeypatch.setattr(
        engine_module.HestonPathSimulator, "simulate_paths", _fail_simulate_paths
    )

    terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=4,
        cfg=MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=23)),
        scheme=scheme,
    )

    assert terminal.shape == (4,)
    assert np.all(np.isfinite(terminal))


@pytest.mark.parametrize("scheme", ["euler_full_truncation", "quadratic_exponential"])
def test_simulate_heston_terminal_matches_full_path_terminal_with_same_seed(
    heston_params: HestonParams,
    scheme: str,
) -> None:
    cfg = MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=37))

    terminal = simulate_heston_terminal(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=cfg,
        scheme=scheme,
    )
    paths = simulate_heston_paths(
        ctx=_ctx(),
        tau=1.0,
        params=heston_params,
        n_steps=8,
        cfg=cfg,
        scheme=scheme,
    )

    np.testing.assert_allclose(terminal, paths[:, -1])
