"""SVI repair / calibration tests.

These tests are intentionally written to be robust to small implementation
changes in the SVI module (e.g. adding a post-repair refit or adding new
regularization terms) while still verifying the key behaviors.
"""

import inspect

import numpy as np
import pytest

import option_pricing.vol.svi as svi


def _arb_bad_params() -> svi.SVIParams:
    """
    Make a raw SVI slice that FAILS butterfly arbitrage via wing-limit violation:
      sR = b(1+rho) > 2  => g_right_inf < 0
    Keep m != 0 and rho != 0 to avoid JW->raw degeneracy edge cases.
    """
    return svi.SVIParams(
        a=0.10,
        b=1.30,
        rho=0.70,
        m=0.10,
        sigma=0.20,
    )


def _arb_good_params() -> svi.SVIParams:
    # Mild, should be butterfly-safe and domain-positive for typical y-ranges.
    return svi.SVIParams(a=0.05, b=0.30, rho=-0.20, m=0.00, sigma=0.25)


def _assert_params_close(
    p: svi.SVIParams, q: svi.SVIParams, *, atol: float = 1e-10
) -> None:
    """Dataclass equality is exact; allow tiny float diffs (e.g. encode/decode roundtrips)."""
    pv = np.array([p.a, p.b, p.rho, p.m, p.sigma], dtype=np.float64)
    qv = np.array([q.a, q.b, q.rho, q.m, q.sigma], dtype=np.float64)
    assert np.allclose(pv, qv, rtol=0.0, atol=atol), (pv, qv)


@pytest.mark.parametrize("method", ["project", "line_search"])
def test_repair_butterfly_raw_returns_feasible(method: str) -> None:
    p_bad = _arb_bad_params()

    b0 = svi.check_butterfly_arbitrage(
        p_bad,
        y_domain_hint=(-1.25, 1.25),
        w_floor=0.0,
        g_floor=0.0,
        tol=1e-10,
    )
    assert not b0.ok, "precondition failed: p_bad should violate butterfly arbitrage"

    p_fix = svi.repair_butterfly_raw(
        p_bad,
        T=1.0,  # you're working in total-variance space
        y_domain_hint=(-1.25, 1.25),
        w_floor=0.0,
        method=method,
        tol=1e-10,
        n_scan=31,
        n_bisect=30,
    )

    b1 = svi.check_butterfly_arbitrage(
        p_fix,
        y_domain_hint=(-1.25, 1.25),
        w_floor=0.0,
        g_floor=0.0,
        tol=1e-10,
    )
    assert b1.ok, f"repair method={method} did not produce a butterfly-safe slice"


def test_line_search_is_not_worse_than_project_in_jw_distance() -> None:
    """
    Sanity: line-search should typically land closer to the original point than full projection.
    (Not mathematically guaranteed in all weird edge-cases, but should hold for this constructed case.)
    """
    p_bad = _arb_bad_params()
    ydom = (-1.25, 1.25)

    p_proj = svi.repair_butterfly_raw(
        p_bad, T=1.0, y_domain_hint=ydom, w_floor=0.0, method="project"
    )
    p_ls = svi.repair_butterfly_raw(
        p_bad, T=1.0, y_domain_hint=ydom, w_floor=0.0, method="line_search"
    )

    jw0 = p_bad.to_jw(T=1.0)
    jw_proj = p_proj.to_jw(T=1.0)
    jw_ls = p_ls.to_jw(T=1.0)

    # Compare only the coordinates you actually change in your repair path: (c, v_tilde)
    d_proj = np.hypot(jw_proj.c - jw0.c, jw_proj.v_tilde - jw0.v_tilde)
    d_ls = np.hypot(jw_ls.c - jw0.c, jw_ls.v_tilde - jw0.v_tilde)

    assert d_ls <= d_proj + 1e-12


def test_calibrate_svi_triggers_repair_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Integration-ish test:
    - Avoid running a real optimizer: monkeypatch least_squares to return x0.
    - Force the first butterfly check in _finalize to fail, so repair runs.
    - Monkeypatch repair_butterfly_raw to return a known good params.
    - Ensure output params are ~repaired ones and diagnostics show butterfly_ok=True.
    """

    class DummyLSQRes:
        def __init__(self, x):
            self.success = True
            self.x = np.asarray(x, dtype=np.float64)
            self.message = "dummy"
            self.nfev = 1
            self.cost = 0.0
            self.optimality = 0.0

    # Patch least_squares used inside your module (robust to signature changes)
    def fake_least_squares(*args, **kwargs):
        if "x0" in kwargs:
            x0 = kwargs["x0"]
        else:
            x0 = args[1]  # least_squares(fun, x0, ...)
        return DummyLSQRes(x0)

    monkeypatch.setattr(svi, "least_squares", fake_least_squares)

    # First butterfly check fails -> triggers repair; later checks succeed
    calls = {"n": 0}

    def fake_check(*args, **kwargs):
        calls["n"] += 1
        ok = calls["n"] != 1
        ydom = kwargs.get("y_domain_hint", (-1.25, 1.25))
        return svi.ButterflyCheck(
            ok=ok,
            y_domain=ydom,
            min_g=0.0 if ok else -1.0,
            argmin_y=0.0,
            n_stationary=0,
            g_left_inf=0.0,
            g_right_inf=0.0,
            stationary_points=tuple(),
            failure_reason=None if ok else "forced_fail",
        )

    monkeypatch.setattr(svi, "check_butterfly_arbitrage", fake_check)

    p_good = _arb_good_params()
    repair_called = {"n": 0}

    def fake_repair(*args, **kwargs):
        repair_called["n"] += 1
        return p_good

    monkeypatch.setattr(svi, "repair_butterfly_raw", fake_repair)

    y = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
    w_obs = np.array([0.08, 0.06, 0.07], dtype=np.float64)

    sig = inspect.signature(svi.calibrate_svi).parameters

    # Optional post-repair refit knobs
    refit_kwargs: dict[str, object] = {}
    if "refit_after_repair" in sig:
        refit_kwargs["refit_after_repair"] = True
        if "refit_max_nfev" in sig:
            refit_kwargs["refit_max_nfev"] = 1

    # Optional g-penalty knobs (disable it for this test if present)
    reg_kwargs: dict[str, object] = {}
    if hasattr(svi, "SVIRegConfig") and "lambda_g" in getattr(
        svi.SVIRegConfig, "__annotations__", {}
    ):
        reg_kwargs["reg_override"] = {"lambda_g": 0.0}

    out = svi.calibrate_svi(
        y=y,
        w_obs=w_obs,
        loss="linear",
        robust_data_only=True,
        repair_butterfly=True,
        x0=_arb_good_params(),
        **refit_kwargs,
        **reg_kwargs,
    )

    assert (
        repair_called["n"] >= 1
    ), "expected calibrate_svi to call repair_butterfly_raw"
    _assert_params_close(out.params, p_good)
    assert out.diag.checks.butterfly_ok is True
